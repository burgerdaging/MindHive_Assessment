import asyncio
import logging
from typing import List, Dict, Any

# Import necessary components from your project structure
from tools.planner import AgenticPlanner
from models.schemas import QueryRequest, QueryResponse # To mimic API request/response
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage # For handling chat history objects

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class TerminalChat:
    def __init__(self):
        # Initialize the AgenticPlanner, which contains the LLM, tools, and memory
        self.planner = AgenticPlanner()
        # This will store the chat history as LangChain BaseMessage objects
        self.chat_history: List[BaseMessage] = [] # Changed type hint to be more precise

    async def handle_query(self, query: str) -> str:
        """
        Processes user query using the AgenticPlanner and returns the response.
        Manages the conversion of chat history between internal LangChain objects
        and Pydantic-compatible dictionaries for the planner.
        """
        try: # <--- UNCOMMENTED THIS 'try' BLOCK
            # 1. Prepare chat history for the planner
            # The planner's `plan_and_execute` expects a list of dictionaries
            # where each dict represents a message (e.g., {"type": "human", "content": "..."})
            # We convert our internal LangChain BaseMessage objects to this format.
            chat_history_for_planner = [msg.dict() for msg in self.chat_history]

            # 2. Create a QueryRequest object (mimics the FastAPI endpoint's input)
            request_data = QueryRequest(
                message=query,
                chat_history=chat_history_for_planner
            )

            # 3. Call the AgenticPlanner's core method
            # This is where the agent decides which tool to use, executes it, and generates a response.
            # It also returns the updated chat history from its internal memory.
            # IMPORTANT: ADD 'await' HERE
            planner_result = await self.planner.plan_and_execute( # <--- ADDED 'await'
                request_data.message,
                request_data.chat_history
            )

            # 4. Extract the final response and the updated chat history
            final_answer = planner_result["final_answer"]
            updated_chat_history_dicts = planner_result["updated_chat_history"]

            # 5. Update the TerminalChat's internal chat_history
            # Convert the dictionary representation back to LangChain BaseMessage objects
            # for consistent internal state and easy printing.
            self.chat_history = []
            for msg_dict in updated_chat_history_dicts:
                if msg_dict['type'] == 'human':
                    self.chat_history.append(HumanMessage(content=msg_dict['content']))
                elif msg_dict['type'] == 'ai':
                    self.chat_history.append(AIMessage(content=msg_dict['content']))
                # Add other message types if your agent uses them (e.g., ToolMessage, FunctionMessage)

            return final_answer

        except Exception as e: # <--- UNCOMMENTED THIS 'except' BLOCK
            error_msg = f"Error processing your request: {str(e)}"
            # Append the error to history for debugging
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=error_msg))
            logger.error(f"Error in handle_query: {e}", exc_info=True) # Log full traceback
            return error_msg

    def print_history(self):
        """Display conversation history using LangChain BaseMessage objects."""
        print("\n=== Conversation History ===")
        if not self.chat_history:
            print("No conversation history yet.")
        else:
            for i, msg in enumerate(self.chat_history):
                if isinstance(msg, HumanMessage):
                    print(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"Bot: {msg.content}")
                # You can add more specific formatting for other message types if needed
        print("===========================")

async def main():
    chat = TerminalChat()
    print("\n" + "="*50)
    print("ZUS Coffee AI Assistant - Terminal Version")
    print("Type 'quit', 'exit', or 'bye' to end the chat")
    print("Type 'history' to view conversation history")
    print("="*50 + "\n")

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() in ['quit', 'exit', 'bye']:
                chat.print_history()
                print("\nGoodbye! Have a great day!\n")
                break

            if query.lower() == 'history':
                chat.print_history()
                continue

            if not query:
                print("Please enter a question or command")
                continue

            response = await chat.handle_query(query)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            chat.print_history()
            print("\nSession ended by user. Goodbye!\n")
            break
        except Exception as e:
            print(f"\nError in main loop: {str(e)}\n")
            logger.error(f"Error in main loop: {e}", exc_info=True)

if __name__ == "__main__":
    # Ensure all necessary environment variables are loaded for the entire system
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.run(main())