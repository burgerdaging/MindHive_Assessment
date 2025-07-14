from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from tools.agent_tools import AVAILABLE_TOOLS
from config import settings
from typing import List, Dict, Any
import logging
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage # Import BaseMessage for type hinting

logger = logging.getLogger(__name__)

class AgenticPlanner:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=settings.GEMINI_API_KEY,
            temperature=0.0
        )
        self.tools = AVAILABLE_TOOLS

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True # This means memory stores BaseMessage objects
        )

        self.agent_executor = self._create_agent()

        logger.info(f"AgenticPlanner initialized with {len(self.tools)} tools:")
        for tool_item in self.tools:
            logger.info(f"  - {tool_item.name}: {tool_item.description[:50]}...")

    def _create_agent(self):
        """Create the agent with enhanced planning capabilities"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent AI assistant specialized in ZUS Coffee information with advanced planning capabilities.\n\n**YOUR TOOLS:**\nYou have access to the following tools:\n- `calculator`: For all mathematical calculations and arithmetic operations.\n- `get_current_time`: To provide the current date and time.\n- `search_zus_products`: To find information about ZUS Coffee drinkware, tumblers, mugs, and other products.\n- `search_zus_outlets`: To find ZUS Coffee store locations, addresses, operating hours, and services.\n- `search_zus_website`: For general ZUS Coffee information not covered by specific product or outlet tools.\n- `knowledge_base_search`: For general factual information not related to ZUS Coffee.\n\n**DECISION PROCESS:**\n1. **Understand Intent**: Carefully analyze the user\'s request.\n2. **Identify Missing Information**: If a tool requires specific details (e.g., numbers for calculation, a specific location for an outlet) and the user hasn\'t provided it, you MUST ask a clarifying question. Do NOT guess.\n3. **Choose Action**: Select the most appropriate tool based on the intent.\n4. **Execute Tool**: Call the chosen tool with the necessary inputs.\n5. **Formulate Answer**: Based on the tool\'s output, provide a clear and concise answer to the user.\n6. **Finish**: If you have fully answered the question, conclude the conversation.\n\n**PLANNING GUIDELINES (PRIORITY ORDER):**\n- **Mathematical Calculations**: If the user asks for *any* arithmetic or calculation, ALWAYS use the `calculator` tool. Do not try to interpret it as a database query.\n- **ZUS Coffee Products/Drinkware**: If the user asks about ZUS Coffee *products*, *drinkware*, *tumblers*, *mugs*, *bottles*, *prices of products*, or where to *buy products*, use `search_zus_products`.\n- **ZUS Coffee Outlets/Stores/Locations**: If the user asks about ZUS Coffee *outlets*, *stores*, *locations*, *addresses*, *branches*, *hours*, or *services* at a store, use `search_zus_outlets`.\n- **General ZUS Coffee Information**: If the user asks a general question about ZUS Coffee (e.g., its history, mission, general offerings) that isn\'t specifically about products or outlets, use `search_zus_website`.\n- **Current Time**: If the user asks for the current time, use `get_current_time`.\n- **General Knowledge**: For any other factual questions not related to ZUS Coffee, use `knowledge_base_search`.\n\n**EXAMPLES FOR CLARIFICATION & MEMORY (Part 1):**\nUser: "Calculate something."\nYou: "I\'d be happy to help with calculations! Could you please provide the specific mathematical expression or numbers you\'d like me to calculate?"\n\nUser: "Is there an outlet in Petaling Jaya?"\nYou: "Yes, there are ZUS Coffee outlets in Petaling Jaya. Which specific outlet are you referring to, or would you like a list of all of them?"\n\nUser: "SS 2, what\'s the opening time?"\nYou: (Uses memory to recall "Petaling Jaya" and "SS 2" and then uses `search_zus_outlets`)\n\n**ERROR HANDLING:**\n- If a tool fails, acknowledge the error and suggest alternatives.\n- Never crash or stop responding.\n- Provide helpful recovery prompts.\n- Maintain conversation flow even during errors.\n"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=settings.DEBUG, # Ensure settings.DEBUG is True for verbose output during debugging
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

    # Corrected plan_and_execute method
    async def plan_and_execute(self, user_input: str, chat_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main planning and execution method that demonstrates the decision-making process.
        Handles conversion of chat history between dictionaries (for external API/main.py)
        and BaseMessage objects (for LangChain memory/agent).
        """
        try:
            logger.info(f"Processing user input: {user_input}")

            # Step 1: Convert incoming chat_history (from main.py) from dicts to BaseMessage objects
            # and set it to the agent's memory.
            # This is crucial because the agent's memory expects BaseMessage objects.
            if chat_history:
                converted_chat_history: List[BaseMessage] = []
                for msg_dict in chat_history:
                    if msg_dict['type'] == 'human':
                        converted_chat_history.append(HumanMessage(content=msg_dict['content']))
                    elif msg_dict['type'] == 'ai':
                        converted_chat_history.append(AIMessage(content=msg_dict['content']))
                    # Add other message types if your agent uses them (e.g., ToolMessage, FunctionMessage)
                self.memory.chat_memory.messages = converted_chat_history
            else:
                # If no chat_history is provided, clear memory for a fresh start (or keep existing)
                # For a terminal chat, memory persists, so this 'else' might not be strictly needed
                # if you always pass the full history from main.py.
                # But it's good for robustness if this method is called stateless.
                self.memory.chat_memory.messages = [] # Or self.memory.clear()

            # Step 2: Execute with agent
            # The agent_executor already has memory, so we just need to pass the current input.
            # The memory will automatically be updated by the agent executor after the turn.
            result = await self.agent_executor.ainvoke({ # Use ainvoke for async execution
                "input": user_input
            })

            # Step 3: Extract decision information
            intermediate_steps = result.get("intermediate_steps", [])
            tools_used = self._extract_tools_used(intermediate_steps)
            decision_points = self._extract_decision_points(intermediate_steps)

            # Step 4: Prepare the updated chat history for return to main.py
            # Convert the BaseMessage objects from the agent's memory back to dictionaries
            # because main.py expects dictionaries.
            updated_chat_history_for_main = [msg.dict() for msg in self.memory.chat_memory.messages]

            planning_info = {
                "user_input": user_input,
                "intent_analysis": self._analyze_intent(user_input), # Call analyze_intent here
                "final_answer": result["output"],
                "tools_used": tools_used,
                "decision_points": decision_points,
                "success": True,
                "debug_info": {
                    "intermediate_steps_count": len(intermediate_steps),
                    "result_keys": list(result.keys())
                },
                "updated_chat_history": updated_chat_history_for_main # <--- CORRECTED: Return list of DICTS
            }

            logger.info(f"Tools used: {tools_used}")
            logger.info(f"Decision points: {decision_points}")
            return planning_info

        except Exception as e:
            logger.error(f"Error in plan_and_execute: {e}", exc_info=True)
            # Ensure updated_chat_history is still a list of dicts even on error
            error_chat_history_for_main = [msg.dict() for msg in self.memory.chat_memory.messages]
            return {
                "user_input": user_input,
                "intent_analysis": "Error in analysis",
                "final_answer": f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question or contact support if the issue persists.",
                "tools_used": [],
                "decision_points": [f"Error occurred: {str(e)}"],
                "success": False,
                "updated_chat_history": error_chat_history_for_main # Still return history as dicts
            }

    def _analyze_intent(self, user_input: str) -> str:
        """Analyze user intent for planning purposes (for logging/debugging)"""
        user_input_lower = user_input.lower()

        # ZUS Coffee specific intents
        if any(word in user_input_lower for word in ['zus', 'coffee']):
            if any(word in user_input_lower for word in ['product', 'drinkware', 'tumbler', 'mug', 'cup', 'bottle', 'price', 'buy', 'shop']):
                return "ZUS Coffee product search required"
            elif any(word in user_input_lower for word in ['outlet', 'store', 'location', 'branch', 'address', 'where', 'hours', 'services']):
                return "ZUS Coffee outlet search required"
            else:
                return "General ZUS Coffee information required"

        # General intents
        elif any(word in user_input_lower for word in ['calculate', 'math', '+', '-', '*', '/', 'equals', 'sum', 'subtract', 'multiply', 'divide']):
            return "Mathematical calculation required"
        elif any(word in user_input_lower for word in ['time', 'date', 'now', 'current']):
            return "Time information required"
        elif any(word in user_input_lower for word in ['what is', 'who is', 'tell me about', 'explain']):
            return "Knowledge retrieval required"
        else:
            return "General conversation"

    def _extract_tools_used(self, intermediate_steps: List) -> List[str]:
        """Extract which tools were used during execution"""
        tools_used = []
        for step in intermediate_steps:
            # Check if step[0] is an AgentAction and has a 'tool' attribute
            if isinstance(step[0], tuple) and len(step[0]) > 0 and hasattr(step[0][0], 'tool'):
                tools_used.append(step[0][0].tool)
            # For newer LangChain versions, intermediate_steps might contain AgentAction objects directly
            elif hasattr(step, 'tool'):
                tools_used.append(step.tool)
        return tools_used

    def _extract_decision_points(self, intermediate_steps: List) -> List[str]:
        """Extract decision points from the agent's reasoning"""
        decision_points = []
        for i, step in enumerate(intermediate_steps):
            # Check if step is a tuple (AgentAction, Observation)
            if isinstance(step, tuple) and len(step) >= 2:
                action = step[0]
                observation = step[1]
                if hasattr(action, 'tool'):
                    decision_points.append(
                        f"Decision {i+1}: Used {action.tool} -> {str(observation)[:100]}..."
                    )
            # For newer LangChain versions, intermediate_steps might contain AgentAction objects directly
            elif hasattr(step, 'tool'):
                decision_points.append(
                    f"Decision {i+1}: Agent thought/acted with {step.tool}..."
                )
        return decision_points