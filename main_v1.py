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


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
        print(f"\nQuery Results: {query_result[:5]}...")
        
        return {"query_result": query_result}
    except Exception as e:
        error_message = f"Error executing query: {str(e)}"
        print(f"\nError: {error_message}")
        return {"query_result": [{"error": error_message}]}

def generate_response(state: GraphState) -> Dict:
    """Generate natural language response from query results."""
    result_sample = state["query_result"][:5]
    total_results = len(state["query_result"])
    
    # Get conversation history from messages
    conversation_history = "\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
        for msg in state["messages"][-4:]  # Last 2 exchanges
        if isinstance(msg, (HumanMessage, AIMessage))
    ])
    
    messages = [
        SystemMessage(content=f"""You are a helpful assistant explaining SQL query results.
            Previous conversation:\n{conversation_history}
            
            Provide clear, concise explanations that focus on the key insights from the data.
            If there's an error, explain what might have gone wrong."""),
        HumanMessage(content=f"""Original question: {state["messages"][-1].content}
            SQL Query used: {state["sql_query"]}
            Sample of results (first 5 of {total_results} total): {result_sample}
            
            Please provide a natural language explanation of these results.""")
    ]
    
    response = llm.invoke(messages)
    print(f"\nGenerated Response: {response.content}")
    
    return {"final_response": response.content}

def create_dvd_rental_graph() -> Graph:
    """Create and configure the DVD rental database query workflow."""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("generate_response", generate_response)
    
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "generate_response")
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
        final_response=""
    )
    
    result = process_dvd_rental_query(initial_state)
    
    print("\nFinal Results:")
    print("=" * 50)
    print(f"Question: {question}")
    print(f"\nSQL Query:\n{result['sql_query']}")
    print(f"\nResponse:\n{result['final_response']}")
