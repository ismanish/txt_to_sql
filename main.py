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

# Define state types
class GraphState(TypedDict):
    messages: List[str]
    sql_query: str
    query_result: List[Dict]
    final_response: str


llm = ChatOpenAI(temperature=0)

# Node for SQL Generation
def generate_sql(state: GraphState) -> GraphState:
    """Generate SQL query from natural language input."""
    inspector = DVDRentalInspector()
    schema_info = inspector.get_schema_for_prompt()
        # Create a comprehensive schema context
    schema_context = (f"Database Schema:\n"
                        f"{schema_info}\n\n"
                        f"Important Rules:\n"
                        f"1. ALWAYS prefix column names with their table name or alias (e.g., customer.customer_id, c.customer_id)\n"
                        f"2. When joining tables:\n"
                        f"   - Use meaningful table aliases (e.g., c for customer, f for film)\n"
                        f"   - Qualify all column references with table aliases\n"
                        f"   - Use proper JOIN syntax (INNER JOIN, LEFT JOIN, etc.)\n"
                        f"3. In GROUP BY and ORDER BY:\n"
                        f"   - Use fully qualified column names\n"
                        f"   - Reference columns by their position number if using expressions\n"
                        f"4. For complex queries, use CTEs (WITH clause) to improve readability\n"
                        f"5. Handle NULL values appropriately using IS NULL or IS NOT NULL\n"
                        f"6. Use DISTINCT when necessary to avoid duplicate rows\n"
                        f"7. Always test edge cases (e.g., no results, NULL values)")    
    example_query = """SELECT 
            c.customer_id,
            c.first_name,
            c.last_name,
            COUNT(r.rental_id) as rental_count
        FROM 
            customer c
            INNER JOIN rental r ON c.customer_id = r.customer_id
        GROUP BY 
            c.customer_id, c.first_name, c.last_name
        ORDER BY 
            rental_count DESC"""
    messages = [
            SystemMessage(content=(f"You are a SQL query generator for a DVD rental database.\n"
                                 f"Generate only the SQL query without any markdown formatting or explanation.\n"
                                 f"The query should be executable in PostgreSQL.\n\n"
                                 f"{schema_context}\n\n"
                                 f"Example of a well-formed query:\n"
                                 f"{example_query}")),
            HumanMessage(content=state["messages"][-1].content)
        ]
        
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(messages)
    sql_query = response.content.strip()
    
    # Clean any potential markdown formatting
    sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
    
    print(f"\nGenerated SQL: {sql_query}")
    
    return {
        "sql_query": sql_query
    }

def execute_sql(state: GraphState) -> Dict:
    """Execute the generated SQL query against the DVD rental database."""
    try:
        conn = psycopg2.connect(**db_config)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(state["sql_query"])
            results = cur.fetchall()
        conn.close()
        
        query_result = [dict(row) for row in results]
        print(f"\nQuery Results: {query_result[:5]}...")  # Print first 5 results
        
        # Return only the modified part of the state
        return {"query_result": query_result}
        
    except Exception as e:
        error_message = f"Error executing query: {str(e)}"
        print(f"\nError: {error_message}")
        return {"query_result": [{"error": error_message}]}


def generate_response(state: GraphState) -> Dict:
    """Generate natural language response from query results."""
    result_sample = state["query_result"][:5]  # Use first 5 results for context
    total_results = len(state["query_result"])
    
    messages = [
        SystemMessage(content="""You are a helpful assistant explaining SQL query results.
            Provide clear, concise explanations that focus on the key insights from the data.
            If there's an error, explain what might have gone wrong."""),
        HumanMessage(content=f"""Original question: {state["messages"][-1].content}
            SQL Query used: {state["sql_query"]}
            Sample of results (first 5 of {total_results} total): {result_sample}
            
            Please provide a natural language explanation of these results.""")
    ]
    
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(messages)
    
    print(f"\nGenerated Response: {response.content}")
    
    # Return only the modified part of the state
    return {"final_response": response.content}


def create_dvd_rental_graph() -> Graph:
    """Create and configure the DVD rental database query workflow."""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("generate_response", generate_response)
    
    # Define edges
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_edge("execute_sql", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Set entry point
    workflow.set_entry_point("generate_sql")
    
    return workflow.compile()


def process_dvd_rental_query(state: GraphState) -> GraphState:
    """Process a natural language query about the DVD rental database."""
    workflow = create_dvd_rental_graph()
    
    # Execute the workflow with provided state
    final_state = workflow.invoke(state)
    
    # Return the final state directly
    return final_state


if __name__ == "__main__":
    question = "What are the top 5 most rated movies in our database?"
    question = "What are the top 5 most rated movies in our database in horror?"
    
    # Create initial state with the question
    initial_state = GraphState(
        messages=[HumanMessage(content=question)],
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