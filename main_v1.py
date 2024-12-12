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
from sql_logger import sql_logger
from decimal import Decimal
import re
from datetime import datetime

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


def extract_relevant_context(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """Extract relevant context from message history.
    Returns a dict with current_question and context."""
    
    user_messages = []
    for message in reversed(messages):
        if message["role"] == "user":
            user_messages.append(message["content"].strip())
            if len(user_messages) >= 2:
                break
                
    if not user_messages:
        return {"current_question": "", "context": ""}
        
    return {
        "current_question": user_messages[0],
        "context": user_messages[1] if len(user_messages) > 1 else ""
    }


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
    try:
        inspector = DVDRentalInspector()
        schema_info = inspector.get_schema_for_prompt()
        
        messages = [
            SystemMessage(content=f"""You are a SQL query analyzer for a DVD rental database.
            Here is the schema information:
            {schema_info}
            
            Analyze the user's question and provide guidance on:
            1. Which tables and columns are needed
            2. Whether it's a title search query
            3. Any specific conditions or joins required
            
            Return your analysis as a JSON with these fields:
            {{
                "tables": ["list of required tables"],
                "columns": ["list of required columns"],
                "is_title_search": boolean,
                "conditions": ["list of conditions needed"],
                "joins": ["list of required joins"]
            }}
            """),
            HumanMessage(content=question)
        ]
        
        response = llm.invoke(messages)
        cleaned_response = clean_json_response(response.content)
        return json.loads(cleaned_response)
        
    except Exception as e:
        print(f"Failed to parse analysis response: {str(e)}")
        return None


def execute_sql(sql_query: str) -> Dict:
    """Execute the generated SQL query against the database."""
    try:
        # Get database configuration
        config = configparser.ConfigParser()
        config.read("database/database.ini")
        db_config = config["local"]
        
        # Connect to the database
        conn = psycopg2.connect(
            host=db_config["host"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"]
        )
        
        # Execute query with dictionary cursor
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_query)
            results = cur.fetchall()
            
            # Convert Decimal objects to float for JSON serialization
            for row in results:
                for key, value in row.items():
                    if isinstance(value, Decimal):
                        row[key] = float(value)
            
            return results
            
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return []
        
    finally:
        if conn:
            conn.close()


def generate_sql(state: GraphState) -> Dict:
    """Generate SQL query based on user input using GPT-4 for analysis."""
    try:
        context = extract_relevant_context(state.messages)
        if not context["current_question"]:
            return {"error": "No user question found"}
            
        inspector = DVDRentalInspector()
        schema_info = inspector.get_schema_for_prompt()
        
        # First generate the SQL query with analysis
        sql_messages = [
            SystemMessage(content=f"""You are a PostgreSQL expert. Generate SQL queries and analyze their correctness.

Database Schema:
{schema_info}

Instructions for Searches:
1. For follow-up questions with "and", treat it as a new independent search
2. Never use IN clause unless explicitly asked for multiple items
3. Each search should be a separate query
4. For "top N per group" queries (e.g., "top N per category/year"):
   - Use window functions with PARTITION BY
   - Example query structure for time-based ranges:
     WITH RankedItems AS (
       SELECT t1.*, 
         ROW_NUMBER() OVER (PARTITION BY [group_column] ORDER BY [order_column] DESC) as rank
       FROM [table] t1
       WHERE [date_column] >= [start_date]
         AND [date_column] <= [end_date]
     )
     SELECT * FROM RankedItems WHERE rank <= N
     ORDER BY [group_column] DESC, rank;
5. Return the query in JSON format:
{{
    "query": "your SQL query here",
    "search_type": "exact_match|pattern_match|other",
    "search_value": "the exact value being searched",
    "is_follow_up": boolean
}}"""),
            
            HumanMessage(content=f"""Previous Question: {context['context']}
Current Question: {context['current_question']}

Generate a SQL query based on the schema and question. Use appropriate table joins and column names from the schema.""")
        ]
        
        sql_response = llm.invoke(sql_messages)
        try:
            response_data = json.loads(clean_json_response(sql_response.content))
            sql_query = response_data.get("query", "")
            
            if not sql_query:
                return {"error": "Failed to generate valid SQL query"}
            
            # For exact match searches, try exact match first
            if response_data.get("search_type") == "exact_match":
                search_value = response_data.get("search_value", "").strip()
                if search_value:
                    # Use exact match based on the table and column from the original query
                    table_match = re.search(r'FROM\s+(\w+)', sql_query, re.IGNORECASE)
                    where_match = re.search(r'WHERE\s+(\w+\.\w+|\w+)\s*=', sql_query, re.IGNORECASE)
                    
                    if table_match and where_match:
                        table = table_match.group(1)
                        column = where_match.group(1)
                        sql_query = f"SELECT * FROM {table} WHERE {column} = '{search_value}'"
            
            # Execute query and check for empty results
            result = execute_sql(sql_query)
            if not result and response_data.get("search_type") == "exact_match":
                state.original_sql = sql_query
                state.sql_query = sql_query
                state.error_messages = ["No exact match found"]
                
                # Trigger pattern matching recovery
                recovery_result = recover_sql_error(state)
                if "error" not in recovery_result:
                    sql_query = recovery_result["sql_query"]
            
        except (json.JSONDecodeError, AttributeError) as e:
            # If parsing fails, extract search parameters from the original query
            sql_query = clean_sql_response(sql_response.content)
            sql_logger.log_step("SQL Generation",
                              error=f"Failed to parse analysis response: {str(e)}")
        
        sql_logger.log_step("Initial SQL Generation",
                          previous_question=context["context"],
                          current_question=context["current_question"],
                          generated_sql=sql_query)
            
        return {
            "sql_query": sql_query
        }

    except Exception as e:
        sql_logger.log_error("SQL Generation", str(e))
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
            
        # Log initial state
        sql_logger.log_step("SQL Error Recovery - Input",
                          original_query=state.sql_query,
                          error_messages=state.error_messages,
                          user_question=user_question)
        
        # Get schema information
        inspector = DVDRentalInspector()
        schema_info = inspector.get_schema_for_prompt()
        
        # Try value pattern recovery
        corrected_query, suggestions = extractor.recover_query(state.sql_query)
        
        sql_logger.log_step("Value Pattern Analysis",
                          suggestions=suggestions,
                          pattern_corrected_query=corrected_query)
        
        # If we have a high confidence match from pattern analysis, use it directly
        if suggestions and 'film.title' in suggestions:
            matches = suggestions['film.title'].get('matches', [])
            if matches and matches[0][1] >= 75:  # If confidence is >= 75%
                best_match = matches[0][0]
                return {
                    "sql_query": f"SELECT f.film_id, f.title, f.release_year FROM film f WHERE f.title = '{best_match}'"
                }
            
        # If no high confidence match, proceed with error analysis
        error_messages = [
            SystemMessage(content=f"""You are a SQL expert helping fix a failed query.
            
            Database Schema:
            {schema_info}
            
            Important Rules:
            1. ALWAYS prefix column names with their table name or alias
            2. Use proper JOIN conditions based on the schema
            3. Handle data types correctly (especially dates and timestamps)
            4. Use appropriate table aliases consistently
            5. Include all necessary columns in GROUP BY
            6. Return ONLY the working SQL query"""),
            
            HumanMessage(content=f"""Question: {user_question}
            Failed Query: {state.sql_query}
            Error: {state.error_messages[0] if state.error_messages else 'Unknown error'}
            Value Pattern Analysis: {json.dumps(suggestions, indent=2) if suggestions else 'No suggestions'}
            
            Analyze the error and suggest specific corrections based on the schema.""")
        ]
        
        # Get error analysis
        error_response = llm.invoke(error_messages)
        
        sql_logger.log_step("Error Analysis",
                          error_analysis=error_response.content)
        
        # Create messages for correction with schema context
        correction_messages = [
            SystemMessage(content=f"""You are a SQL expert generating a corrected query.
            
            Database Schema:
            {schema_info}
            
            Rules for Query Generation:
            1. Use exact column names from the schema
            2. Include proper table aliases (e.g., f for film, c for customer)
            3. Join tables using their relationship keys
            4. Handle data types appropriately
            5. Use CTEs for complex queries
            6. Ensure proper aggregation and GROUP BY
            7. Return ONLY the SQL query without explanations"""),
            
            HumanMessage(content=f"""Question: {user_question}
            Failed Query: {state.sql_query}
            Error Analysis: {error_response.content}
            
            Generate a corrected SQL query using the proper schema.""")
        ]
        
        correction_response = llm.invoke(correction_messages)
        corrected_query = clean_sql_response(correction_response.content)
        
        sql_logger.log_step("First Recovery Attempt",
                          corrected_query=corrected_query)
        
        return {
            "sql_query": corrected_query
        }
        
    except Exception as e:
        sql_logger.log_error("SQL Recovery", str(e))
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
            
        # Get schema information
        inspector = DVDRentalInspector()
        schema_info = inspector.get_schema_for_prompt()
            
        sql_logger.log_step("Final SQL Correction - Input",
                          user_question=user_question,
                          failed_query=state.sql_query,
                          error_history=state.error_messages)
        
        # Create messages for final correction attempt with schema
        final_messages = [
            SystemMessage(content=f"""You are a SQL expert making a final attempt to fix a failed query.
            
            Database Schema:
            {schema_info}
            
            Critical Rules:
            1. Use EXACT column names from the schema
            2. Join tables ONLY on their correct relationship keys
            3. Handle all data types properly
            4. Use appropriate table aliases consistently
            5. Include all necessary columns in GROUP BY
            6. Return ONLY the working SQL query
            
            Common Fixes:
            1. For date/time operations, use proper PostgreSQL date functions
            2. For rankings, use window functions with correct PARTITION BY
            3. For aggregations, ensure GROUP BY includes all non-aggregated columns
            4. For joins, follow the exact relationships in the schema"""),
            
            HumanMessage(content=f"""Question: {user_question}
            Failed Query: {state.sql_query}
            Error History: {json.dumps(state.error_messages, indent=2)}
            
            This is the final attempt. Generate a working SQL query using the exact schema.""")
        ]
        
        # Generate final correction
        final_response = llm.invoke(final_messages)
        final_query = clean_sql_response(final_response.content)
        
        sql_logger.log_step("Final SQL Correction - Output",
                          final_query=final_query)
        
        return {
            "sql_query": final_query
        }
        
    except Exception as e:
        sql_logger.log_error("Final SQL Correction", str(e))
        return {"error": str(e)}


def generate_response(state: GraphState) -> Dict:
    """Generate a natural language response based on the SQL query and results."""
    try:
        if not state.query_result:
            return {
                "final_response": "No results found for your query."
            }
            
        # Get the last user question
        user_question = ""
        for message in reversed(state.messages):
            if message["role"] == "user":
                user_question = message["content"]
                break
                
        # Convert datetime objects to string in query results
        serializable_results = []
        for row in state.query_result:
            serialized_row = {}
            for key, value in row.items():
                if isinstance(value, datetime):
                    serialized_row[key] = value.isoformat()
                else:
                    serialized_row[key] = value
            serializable_results.append(serialized_row)
                
        # Create messages for response generation
        response_messages = [
            SystemMessage(content="""You are a helpful database assistant. Generate responses following these rules:
1. Present ALL results in a clear, organized format
2. Group results by year when dealing with yearly data
3. Include all relevant fields from the results
4. Show EVERY result returned by the query, do not skip any
5. For each year in the results, show all movies for that year
6. Format using markdown for better readability
7. If showing multiple years, use headers for each year
8. Don't say 'not found in database' if you have valid results
9. Focus on presenting the actual query results"""),
            
            HumanMessage(content=f"""Question: {user_question}
SQL Query: {state.sql_query}
Query Results: {json.dumps(serializable_results, indent=2)}

Generate a clear response that presents ALL the results in an organized way.""")
        ]
        
        response = llm.invoke(response_messages)
        return {
            "final_response": response.content
        }
        
    except Exception as e:
        sql_logger.log_error("Response Generation", str(e))
        return {"error": str(e)}


def process_dvd_rental_query(state: GraphState) -> Dict:
    """Process a DVD rental database query through multiple attempts if needed."""
    try:
        # Reset logger status for new query
        sql_logger.reset_status()
        
        # Generate initial SQL
        result = generate_sql(state)
        if "error" in result:
            return {"error": result["error"]}
            
        sql_query = result["sql_query"]
        state.original_sql = sql_query
        state.sql_query = sql_query
        
        # Execute SQL query
        try:
            query_result = execute_sql(sql_query)
            state.query_result = query_result
            
            # Check for empty results and try pattern recovery
            if not query_result or (isinstance(query_result, list) and len(query_result) == 0):
                sql_logger.log_step("Empty Results - Attempting Pattern Recovery",
                                  original_query=sql_query)
                                  
                # Try value pattern recovery
                corrected_query, suggestions = extractor.recover_query(sql_query)
                
                if suggestions:
                    sql_logger.log_step("Value Pattern Analysis",
                                      suggestions=suggestions,
                                      pattern_corrected_query=corrected_query)
                    
                    # Execute recovered query
                    recovered_result = execute_sql(corrected_query)
                    if recovered_result and len(recovered_result) > 0:
                        state.sql_query = corrected_query
                        state.query_result = recovered_result
                        
                        # Generate response with suggestions
                        response_result = generate_response(state)
                        return {
                            "sql_query": corrected_query,
                            "query_result": recovered_result,
                            "suggestions": suggestions,
                            "final_response": response_result.get("final_response", 
                                "Found similar matches based on your search.")
                        }
            
            # Log successful execution
            sql_logger.log_step("SQL Execution",
                              sql_query=state.sql_query,
                              result="Success - Query executed without errors")
            
            # Generate final response
            response_result = generate_response(state)
            if "error" in response_result:
                return {
                    "sql_query": state.sql_query,
                    "query_result": state.query_result,
                    "final_response": "Successfully retrieved the results, but had trouble generating a natural response."
                }
            
            return {
                "sql_query": state.sql_query,
                "query_result": state.query_result,
                "final_response": response_result["final_response"]
            }
            
        except Exception as e:
            # Log execution error
            sql_logger.log_error("SQL Execution", str(e))
            
            # Try to recover from error
            recovery_result = recover_sql_error(state)
            if recovery_result.get("success"):
                # Generate response for recovered query
                response_result = generate_response(state)
                return {
                    "sql_query": state.sql_query,
                    "query_result": state.query_result,
                    "final_response": response_result.get("final_response", "Successfully retrieved the results after recovery.")
                }
                
            # If recovery failed, try final correction
            correction_result = final_sql_correction(state)
            if correction_result.get("success"):
                # Generate response for corrected query
                response_result = generate_response(state)
                return {
                    "sql_query": state.sql_query,
                    "query_result": state.query_result,
                    "final_response": response_result.get("final_response", "Successfully retrieved the results after final correction.")
                }
                
            return {
                "error": "Failed to generate valid SQL query after multiple attempts",
                "final_response": "I apologize, but I'm having trouble generating a working SQL query. Could you please rephrase your question?"
            }
            
    except Exception as e:
        sql_logger.log_error("Query Processing", str(e))
        return {
            "error": f"Error processing query: {str(e)}",
            "final_response": "An error occurred while processing your query. Could you please try again?"
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