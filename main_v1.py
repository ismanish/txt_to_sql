import os
from typing import Annotated, Sequence, TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain_openai import ChatOpenAI
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph, END
from db_inspector import DVDRentalInspector
import configparser
from query_patterns import ValuePatternExtractor
import json

load_dotenv()

# Load database configuration
config = configparser.ConfigParser()
config.read("database/database.ini")
db_config = dict(config["local"])

class GraphState:
    """State object for managing conversation and query execution state."""
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
        self.sql_query: str = ""
        self.original_sql: str = ""
        self.query_result: List = []
        self.error_messages: List[str] = []
        self.suggestions: Optional[List[str]] = None

    def get(self, key: str, default=None):
        """Get a state attribute with a default value."""
        return getattr(self, key, default)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize query pattern extractor
columns_to_check = [
    ("film", "title"),
    ("customer", "first_name"),
    ("customer", "last_name"),
    ("category", "name")
]
extractor = ValuePatternExtractor(columns_to_check, db_config)


def trim_messages(messages: List[Dict[str, str]], max_messages: int = 5) -> List[Dict[str, str]]:
    """Trim message history to last N messages to avoid token limits."""
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def extract_relevant_context(messages: List[Dict[str, str]]) -> str:
    """Extract relevant context from message history."""
    # Get the last user message
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return ""


def clean_sql_response(response: str) -> str:
    """Clean SQL response by removing markdown formatting and extra text."""
    # Extract SQL query from the response
    lines = response.split('\n')
    sql_lines = []
    in_sql_block = False
    sql_keywords = ['SELECT', 'WITH', 'FROM', 'JOIN', 'WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT', 'HAVING']
    
    # First try to find SQL within code blocks
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Handle SQL code block markers
        if line.startswith('```'):
            # Only switch state if it's a SQL block
            if line.lower().startswith('```sql'):
                in_sql_block = True
            elif line == '```':
                in_sql_block = False
            continue
            
        # Collect lines within SQL blocks or that look like SQL
        if in_sql_block or any(line.upper().startswith(kw) for kw in sql_keywords):
            # Remove any leading comments or explanations
            if any(line.upper().startswith(kw) for kw in sql_keywords):
                sql_lines.append(line)
            elif line.strip().startswith('--') or line.strip().startswith('/*'):
                continue
            else:
                sql_lines.append(line)
    
    # If no SQL found in code blocks, look for SQL-like statements
    if not sql_lines:
        for line in lines:
            line = line.strip()
            if any(line.upper().startswith(kw) for kw in sql_keywords):
                sql_lines.append(line)
                # Continue collecting lines until we hit a line that looks like text
                continue
            elif sql_lines and (line.endswith(';') or any(kw in line.upper() for kw in sql_keywords)):
                sql_lines.append(line)
    
    sql = ' '.join(sql_lines)
    # Remove any trailing semicolons
    sql = sql.strip().rstrip(';')
    
    return sql


def clean_json_response(response: str) -> str:
    """Clean and prepare response for JSON parsing."""
    # Remove any markdown formatting
    cleaned = response.replace('```json', '').replace('```', '')
    # Remove any leading/trailing whitespace
    cleaned = cleaned.strip()
    # If response starts with a newline and quotes, clean it up
    if cleaned.startswith('\\n'):
        cleaned = cleaned[2:]
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    return cleaned


def analyze_schema_for_query(question: str) -> dict:
    """
    Use GPT-4 to analyze the schema and determine the best approach for the query.
    Returns a dictionary with the analysis.
    """
    schema_analyzer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    schema_prompt = """Given a DVD rental database with these EXACT table names:
- film (NOT films)
- actor (NOT actors)
- film_actor
- category
- film_category
- rental
- inventory

Analyze this question and determine:
1. Which tables and joins are needed
2. What metrics or calculations are required
3. How to best measure concepts like 'popularity' or 'best' based on available data

Question: {question}

Return a JSON object in this exact format (no other text):
{{
    "required_tables": ["list of tables needed"],
    "metrics": ["list of metrics to calculate"],
    "joins": ["list of necessary joins"],
    "ordering": "how to order results",
    "reasoning": "brief explanation of approach"
}}"""

    messages = [
        SystemMessage(content="You are a database expert that analyzes queries and determines the optimal approach. Return only valid JSON."),
        HumanMessage(content=schema_prompt.format(question=question))
    ]
    
    try:
        response = schema_analyzer.invoke(messages)
        # Clean and parse the JSON response
        cleaned_response = clean_json_response(response.content)
        analysis = json.loads(cleaned_response)
        return analysis
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response: {response.content}")
        # Return a basic structure
        return {
            "required_tables": ["film", "actor", "film_actor", "inventory", "rental"],
            "metrics": ["rental_count"],
            "joins": ["film to film_actor to actor", "film to inventory to rental"],
            "ordering": "rental_count DESC",
            "reasoning": "Count rentals per film to measure popularity"
        }


def execute_sql(sql_query: str) -> Dict:
    """Execute the generated SQL query against the database."""
    try:
        conn = psycopg2.connect(**db_config)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_query)
            results = cur.fetchall()
        conn.close()
        
        query_result = [dict(row) for row in results]
        print(f"\nSQL Query: {sql_query}")
        print(f"Query Results: {query_result}")
        
        return query_result
    except Exception as e:
        error_message = f"Error executing query: {str(e)}"
        print(f"\nError: {error_message}")
        return {"error": error_message}


def generate_sql(state: GraphState) -> Dict:
    """Generate SQL query based on user input using GPT-4 for analysis."""
    try:
        # Get the last user question
        user_question = ""
        for message in reversed(state.messages):
            if message["role"] == "user":
                user_question = message["content"]
                break
                
        if not user_question:
            return {"error": "No user question found"}

        # Create messages for SQL generation
        sql_messages = [
            SystemMessage(content="""You are a SQL expert. Generate a PostgreSQL query to answer the user's question.
            Use these EXACT table names:
            - film (for movies)
            - actor (for actors)
            - film_actor (joins films and actors)
            - category (for genres)
            - film_category (joins films and categories)
            - rental (for rental history)
            - inventory (for stock)
            
            Rules:
            1. Return ONLY the SQL query, no explanations
            2. Use proper table aliases
            3. Include proper JOIN conditions
            4. Handle any needed aggregations
            5. When ranking movies by popularity:
               - First count rentals per film using COUNT(r.rental_id)
               - Use DENSE_RANK() or ROW_NUMBER() to rank movies
               - Show ALL actors for each ranked movie
            6. Use rental_count or rental_rate for popularity
            7. ALWAYS use DISTINCT ON or appropriate window functions to show all unique combinations
            8. ALWAYS respect the number of results requested (e.g., top 10, top 5)
            9. If no limit is specified, default to showing ALL results"""),
            HumanMessage(content=user_question)
        ]
        
        # Generate SQL query
        sql_response = llm.invoke(sql_messages)
        sql_query = clean_sql_response(sql_response.content)
        
        if not sql_query:
            return {"error": "Failed to generate valid SQL query"}
            
        print("\nGenerated SQL Query:")
        print(sql_query)
        print()
            
        return {
            "sql_query": sql_query
        }
        
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return {"error": str(e)}

def recover_sql_error(state: GraphState) -> Dict:
    """Attempt to recover from SQL execution errors."""
    try:
        # Get the last user question
        user_question = ""
        for message in reversed(state.messages):
            if message["role"] == "user":
                user_question = message["content"]
                break
                
        if not user_question:
            return {"error": "No user question found"}
            
        # Create messages for error analysis
        error_messages = [
            SystemMessage(content="""You are a SQL expert helping fix a failed query.
            The database has tables: film, actor, category, film_actor, film_category, rental, customer, payment.
            Analyze the error and suggest corrections."""),
            HumanMessage(content=f"""Question: {user_question}
            Failed Query: {state.sql_query}
            Error: {state.error_messages[0] if state.error_messages else 'Unknown error'}
            
            What corrections are needed?""")
        ]
        
        # Get error analysis
        error_response = llm.invoke(error_messages)
        
        # Create messages for correction
        correction_messages = [
            SystemMessage(content="""You are a SQL expert. Generate a corrected SQL query.
            Use exact table names: film, actor, category, film_actor, film_category, rental, customer, payment.
            Ensure proper table names and join conditions."""),
            HumanMessage(content=f"""Question: {user_question}
            Failed Query: {state.sql_query}
            Error Analysis: {error_response.content}
            
            Generate a corrected SQL query.""")
        ]
        
        correction_response = llm.invoke(correction_messages)
        corrected_query = clean_sql_response(correction_response.content)
        
        return {
            "sql_query": corrected_query
        }
        
    except Exception as e:
        return {"error": str(e)}

def final_sql_correction(state: GraphState) -> Dict:
    """Final attempt to correct SQL query after multiple failures."""
    try:
        # Get the last user question
        user_question = ""
        for message in reversed(state.messages):
            if message["role"] == "user":
                user_question = message["content"]
                break
                
        if not user_question:
            return {"error": "No user question found"}
            
        # Create messages for final correction
        messages = [
            SystemMessage(content="""You are a SQL expert making a final attempt to fix a query.
            The database has tables: film, actor, category, film_actor, film_category, rental, customer, payment.
            Generate a simple, reliable query that will definitely work."""),
            HumanMessage(content=f"""Question: {user_question}
            Previous Errors: {', '.join(state.error_messages)}
            
            Generate a simple, reliable SQL query that will work.""")
        ]
        
        # Generate final SQL attempt
        response = llm.invoke(messages)
        final_query = clean_sql_response(response.content)
        
        # Try to execute it
        try:
            result = execute_sql(final_query)
            return {
                "sql_query": final_query,
                "query_result": result
            }
        except Exception as e:
            return {
                "error": str(e),
                "sql_query": final_query,
                "suggestions": ["Try asking about specific tables or columns",
                              "Use simpler conditions in your question",
                              "Specify exactly what information you want"]
            }
            
    except Exception as e:
        return {"error": str(e)}

def generate_response(state: GraphState) -> Dict:
    """Generate a natural language response based on the SQL query and results."""
    try:
        # Get the last user question
        user_question = ""
        for message in reversed(state.messages):
            if message["role"] == "user":
                user_question = message["content"]
                break
                
        if not user_question:
            return {"error": "No user question found"}
            
        # Create messages for response generation
        messages = [
            SystemMessage(content="You are a helpful assistant explaining SQL query results in natural language."),
            HumanMessage(content=f"""Question: {user_question}
            SQL Query: {state.sql_query}
            Query Result: {state.query_result}
            
            Explain these results in a clear, natural way.""")
        ]
        
        # Generate natural language response
        response = llm.invoke(messages)
        
        return {
            "final_response": response.content
        }
        
    except Exception as e:
        return {"error": str(e)}

def process_dvd_rental_query(state: GraphState) -> Dict:
    """Process a DVD rental database query through multiple attempts if needed."""
    try:
        # First attempt with original query
        result = generate_sql(state)
        if "error" in result:
            raise Exception(result["error"])
            
        state.sql_query = result["sql_query"]
        state.original_sql = result["sql_query"]  # Store original query
        
        try:
            state.query_result = execute_sql(state.sql_query)
            if isinstance(state.query_result, dict) and "error" in state.query_result:
                raise Exception(state.query_result["error"])
                
            response_result = generate_response(state)
            if "error" in response_result:
                raise Exception(response_result["error"])
                
            return {
                "sql_query": state.sql_query,
                "query_result": state.query_result,
                "final_response": response_result["final_response"]
            }
        except Exception as e:
            # First SQL execution failed, try recovery
            state.error_messages = [str(e)]
            recovery_result = recover_sql_error(state)
            
            if "error" in recovery_result:
                # Recovery failed, try final correction
                state.error_messages.append(recovery_result["error"])
                final_result = final_sql_correction(state)
                
                if "error" in final_result:
                    # All attempts failed
                    return {
                        "sql_query": final_result.get("sql_query", ""),
                        "query_result": [],
                        "final_response": "I apologize, but I'm having trouble generating a working SQL query. Could you please rephrase your question?",
                        "suggestions": final_result.get("suggestions", ["Try being more specific about what you're looking for."])
                    }
                
                # Final correction worked
                state.sql_query = final_result["sql_query"]
                state.query_result = final_result["query_result"]
                response_result = generate_response(state)
                
                return {
                    "sql_query": state.sql_query,
                    "query_result": state.query_result,
                    "final_response": response_result.get("final_response", "Successfully retrieved the results, but had trouble generating a natural response.")
                }
            
            # Recovery worked, try executing recovered query
            try:
                state.sql_query = recovery_result["sql_query"]
                state.query_result = execute_sql(state.sql_query)
                if isinstance(state.query_result, dict) and "error" in state.query_result:
                    raise Exception(state.query_result["error"])
                    
                response_result = generate_response(state)
                
                return {
                    "sql_query": state.sql_query,
                    "query_result": state.query_result,
                    "final_response": response_result.get("final_response", "Successfully retrieved the results."),
                    "suggestions": recovery_result.get("suggestions")
                }
            except Exception as e:
                # Recovery execution failed, try final correction
                state.error_messages.append(str(e))
                final_result = final_sql_correction(state)
                
                if "error" in final_result:
                    return {
                        "sql_query": final_result.get("sql_query", ""),
                        "query_result": [],
                        "final_response": "I apologize, but I'm having trouble generating a working SQL query. Could you please rephrase your question?",
                        "suggestions": final_result.get("suggestions", ["Try being more specific about what you're looking for."])
                    }
                
                state.sql_query = final_result["sql_query"]
                state.query_result = final_result["query_result"]
                response_result = generate_response(state)
                
                return {
                    "sql_query": state.sql_query,
                    "query_result": state.query_result,
                    "final_response": response_result.get("final_response", "Successfully retrieved the results, but had trouble generating a natural response.")
                }
                
    except Exception as e:
        return {
            "sql_query": "",
            "query_result": [],
            "final_response": f"I encountered an error: {str(e)}. Could you please rephrase your question?",
            "suggestions": ["Try being more specific about what you're looking for."]
        }

if __name__ == "__main__":
    question = "What are the top 5 most rated movies in our database?"
    
    # Create initial state with the question
    initial_state = GraphState()
    initial_state.messages = [
        SystemMessage(content="You are a helpful assistant for querying a DVD rental database."),
        HumanMessage(content=question)
    ]
    
    result = process_dvd_rental_query(initial_state)
    
    print("\nFinal Results:")
    print("=" * 50)
    print(f"Question: {question}")
    print(f"\nSQL Query:\n{result['sql_query']}")
    print(f"\nResponse:\n{result['final_response']}")

samply_question  = "show top rated movies 2 for each year for last 5 years"