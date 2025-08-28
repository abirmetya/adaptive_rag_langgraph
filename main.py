from dotenv import load_dotenv

load_dotenv()

from graph.graph import workflow

def main():
    print("="*60)
    print("Advanced RAG Chatbot")
    print("="*60)
    print("Welcome! Ask me anything or type 'quit', 'exit', or 'bye' to stop.")
    print("-"*60)

    while True:
        try:
            # Get user input
            user_question = input("\n You: ").strip()
            
            # Check for exit commands
            if user_question.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n Goodbye! Thanks for chatting!")
                break

            # Skip empty inputs
            if not user_question:
                print("Please enter a question.")
                continue

            # Show processing indicator
            print("\n Bot: Thinking...")

            # Process the question through the graph
            result = workflow.invoke({"query": user_question})

            print(f"\n Bot: {result}")

        except KeyboardInterrupt:
            print("\n\n Goodbye! Thanks for chatting!")
        except Exception as e:
            print(f"\n Sorry, I encountered an error: {str(e)}")
            print("Please try asking your question again")

if __name__ == "__main__":
    main()
