import os
from typing import List
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from main import process_dvd_rental_query, GraphState

def print_welcome():
    """Print welcome message."""
    print("\nDVD Rental Database Chat Assistant")
    print("Type 'q' to quit\n")

def main():
    messages = [
        SystemMessage(content="You are a helpful assistant for querying a DVD rental database.")
    ]
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'q':
            break
            
        try:
            # Create proper GraphState
            state = GraphState(
                messages=messages + [HumanMessage(content=user_input)],
                sql_query="",
                query_result=[],
                final_response=""
            )
            
            result = process_dvd_rental_query(state)
            print("\nAssistant:", result["final_response"]) 
            
            messages.append(HumanMessage(content=user_input))
            messages.append(AIMessage(content=result["final_response"]))  
            if len(messages) > 10:
                messages = [messages[0]] + messages[-9:]
                
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    print_welcome()
    main()