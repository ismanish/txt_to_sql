import os
from typing import List
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from main_v1 import process_dvd_rental_query, GraphState

def format_suggestions(suggestions):
    """Format suggestions into a readable string."""
    if not suggestions:
        return ""
        
    formatted = "\nSuggestions for query correction:"
    for column_key, data in suggestions.items():
        formatted += f"\nFor {column_key}:"
        formatted += f"\n  Original value: '{data['original']}'"
        formatted += f"\n  Suggested matches:"
        for match, score in data['matches']:
            formatted += f"\n    - '{match}' (confidence: {score}%)"
    return formatted

def print_welcome():
    """Print welcome message."""
    print("\nDVD Rental Database Chat Assistant")
    print("Type 'q' to quit")
    print("This version remembers context from previous questions!")
    print("\nExample follow-up questions you can try:")
    print("1. Ask about movies, then ask 'Which of these had the highest rental rate?'")
    print("2. Ask about customers, then ask 'How many rentals did they make last month?'")
    print()

def main():
    # Initialize conversation with system message
    messages = [
        SystemMessage(content="You are a helpful assistant for querying a DVD rental database.")
    ]
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'q':
            print("\nGoodbye! Thanks for chatting!")
            break
            
        try:
            # Add user message to history
            messages.append(HumanMessage(content=user_input))
            
            # Create state with full message history
            state = GraphState(
                messages=messages,
                sql_query="",
                query_result=[],
                final_response="",
                suggestions=None
            )
            
            result = process_dvd_rental_query(state)
            
            # Add assistant's response to history
            messages.append(AIMessage(content=result["final_response"]))
            
            # Print assistant's response
            print("\nAssistant:", result["final_response"])
            
            # Display suggestions if any were generated
            if result.get("suggestions"):
                print(format_suggestions(result["suggestions"]))
            
            # Keep only recent history (last 5 exchanges + system message)
            if len(messages) > 11:  # 1 system message + 5 pairs of messages
                messages = [messages[0]] + messages[-10:]
                
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    print_welcome()
    main()
