import os
from typing import Annotated, Sequence, TypedDict, Optional, List, Dict, Any
from dotenv import load_dotenv
import psycopg2
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import Graph, StateGraph, END
from db_inspector import DVDRentalInspector
from psycopg2.extras import RealDictCursor
import configparser
from query_patterns import ValuePatternExtractor
import json

load_dotenv()

config_file = "database/database.ini"
env = 'local'

config = configparser.ConfigParser()
config.read(config_file)
db_config = config[env]

# Define state types with memory
class GraphState(TypedDict):
    messages: List[Any]  # List of messages with memory
    sql_query: str
    query_result: List[Dict]
    final_response: str
    suggestions: Optional[Dict[str, Any]]  # Store query correction suggestions


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize query pattern extractor
columns_to_check = [
    ("film", "title"),
    ("customer", "first_name"),
    ("customer", "last_name"),
    ("category", "name")
]
extractor = ValuePatternExtractor(columns_to_check, db_config)


def generate_sql(state: GraphState) -> GraphState:
    """Generate SQL query from natural language input."""
    inspector = DVDRentalInspector()
    schema_info = inspector.get_schema_for_prompt()
    
    # Create a comprehensive schema context
    schema_context = (f"Database Schema:\n"
                     f"{schema_info}\n\n"
                     f"Important Rules:\n"
                     f"1. ALWAYS prefix column names with their table name or alias\n"
                     f"2. Use proper JOIN syntax and table aliases\n"
                     f"3. Handle NULL values appropriately\n"
                     f"4. Use DISTINCT when necessary")

    # Get conversation history from messages
    conversation_history = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
        for msg in state["messages"][-4:]  # Last 2 exchanges
        if isinstance(msg, (HumanMessage, AIMessage))
    ])
    
    messages = [
        SystemMessage(content=(f"You are a SQL query generator for a DVD rental database.\n"
                             f"Generate only the SQL query without any markdown formatting.\n"
                             f"{schema_context}\n\n"
                             f"Previous conversation:\n{conversation_history}")),
        HumanMessage(content=state["messages"][-1].content)
    ]
    
    response = llm.invoke(messages)
    sql_query = response.content.strip().replace('```sql', '').replace('```', '').strip()
    
    print(f"\nGenerated SQL: {sql_query}")
    
    return {"sql_query": sql_query}


def execute_sql(state: GraphState) -> Dict:
    """Execute the generated SQL query against the database."""
    try:
        conn = psycopg2.connect(**db_config)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(state["sql_query"])
            results = cur.fetchall()
        conn.close()
        
        query_result = [dict(row) for row in results]
        print(f"\nQuery Results: {query_result}")  # Print all results
        
        return {"query_result": query_result, "suggestions": None}
    except Exception as e:
        error_message = f"Error executing query: {str(e)}"
        print(f"\nError: {error_message}")
        return {"error": error_message}


def recover_sql(state: GraphState) -> Dict:
    """Attempt to recover from SQL execution errors."""
    if "error" not in state:
        return state
        
    print("\nAttempting to recover from SQL error...")
    corrected_query, suggestions = extractor.recover_query(state["sql_query"])
    
    if corrected_query != state["sql_query"]:
        print(f"\nCorrected SQL: {corrected_query}")
        try:
            conn = psycopg2.connect(**db_config)
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(corrected_query)
                results = cur.fetchall()
            conn.close()
            
            query_result = [dict(row) for row in results]
            print(f"\nRecovered Query Results: {query_result[:5]}...")
            
            return {
                "sql_query": corrected_query,
                "query_result": query_result,
                "suggestions": suggestions
            }
        except Exception as e:
            print(f"\nError executing recovered query: {str(e)}")
            return state
    
    return state


def generate_response(state: GraphState) -> Dict:
    """Generate a natural language response based on the SQL query and results."""
    
    # Create a prompt that includes the query and results
    result_str = ""
    if state["query_result"]:
        # Convert each row to a dictionary for better formatting
        result_dicts = [dict(row) for row in state["query_result"]]
        result_str = json.dumps(result_dicts, indent=2)
    
    prompt = f"""Given the following SQL query and its results, generate a natural language response that clearly explains the findings.
    
SQL Query: {state["sql_query"]}

Query Results: {result_str}

Please provide a clear and complete response that mentions all results. If there are multiple items, list them all.
If there are no results, clearly state that nothing was found matching the criteria."""

    # Generate response using chat model
    chat = ChatOpenAI(temperature=0)
    messages = [
        SystemMessage(content="You are a helpful database assistant that provides clear and complete responses."),
        HumanMessage(content=prompt)
    ]
    
    response = chat.invoke(messages)
    return {"final_response": response.content}


def create_dvd_rental_graph() -> Graph:
    """Create and configure the DVD rental database query workflow."""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("recover_sql", recover_sql)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges with conditional routing
    workflow.add_edge("generate_sql", "execute_sql")
    
    # Add conditional edges based on execution result
    workflow.add_conditional_edges(
        "execute_sql",
        lambda x: "recover_sql" if "error" in x else "generate_response"
    )
    
    # Connect recovery node back to response
    workflow.add_edge("recover_sql", "generate_response")
    workflow.add_edge("generate_response", END)
    
    workflow.set_entry_point("generate_sql")
    
    return workflow.compile()


def process_dvd_rental_query(state: GraphState) -> GraphState:
    """Process a natural language query about the DVD rental database."""
    workflow = create_dvd_rental_graph()
    final_state = workflow.invoke(state)
    return final_state


if __name__ == "__main__":
    question = "What are the top 5 most rated movies in our database?"
    
    # Create initial state with the question
    initial_state = GraphState(
        messages=[
            SystemMessage(content="You are a helpful assistant for querying a DVD rental database."),
            HumanMessage(content=question)
        ],
        sql_query="",
        query_result=[],
        final_response="",
        suggestions=None
    )
    
    result = process_dvd_rental_query(initial_state)
    
    print("\nFinal Results:")
    print("=" * 50)
    print(f"Question: {question}")
    print(f"\nSQL Query:\n{result['sql_query']}")
    print(f"\nResponse:\n{result['final_response']}")
