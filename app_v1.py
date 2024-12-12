from typing import Dict, List, Optional
import sys
from main_v1 import process_dvd_rental_query, GraphState

def print_welcome():
    print("\nDVD Rental Database Chat Assistant")
    print("Type 'q' to quit")
    print("This version remembers context from previous questions!")
    print("\nExample follow-up questions you can try:")
    print("1. Ask about movies, then ask 'Which of these had the highest rental rate?'")
    print("2. Ask about customers, then ask 'How many rentals did they make last month?'")
    print("\n")

def main():
    print_welcome()
    
    # Initialize conversation state
    conversation_state = GraphState()
    conversation_state.messages.append({
        "role": "system",
        "content": "You are a helpful assistant for querying a DVD rental database."
    })
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\nGoodbye!")
                break
                
            # Add user message to conversation history
            conversation_state.messages.append({
                "role": "user", 
                "content": user_input
            })
            
            # Process the query
            result = process_dvd_rental_query(conversation_state)
            
            # Add assistant response to conversation history
            if result.get("final_response"):
                conversation_state.messages.append({
                    "role": "assistant",
                    "content": result["final_response"]
                })
                print("\nAssistant:", result["final_response"])
            
            # Print suggestions if any
            if result.get("suggestions"):
                print("\nSuggestions:")
                for suggestion in result["suggestions"]:
                    print(f"- {suggestion}")
                    
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    main()
