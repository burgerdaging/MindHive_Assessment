from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from tools.agent_tools import AVAILABLE_TOOLS  # Import your tools
from config import settings
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class AgenticPlanner:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            api_key=settings.GEMINI_API_KEY, 
            temperature=0.0
        )
        self.tools = AVAILABLE_TOOLS  # This connects your tools
        
        # ADD MEMORY HERE 
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.agent_executor = self._create_agent()
        
        # Log available tools for debugging
        logger.info(f"AgenticPlanner initialized with {len(self.tools)} tools:")
        for tool in self.tools:
            logger.info(f"  - {tool.name}: {tool.description[:50]}...")
        
    def _create_agent(self):
        """Create the agent with enhanced planning capabilities"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an intelligent AI assistant specialized in ZUS Coffee information with advanced planning capabilities.

**YOUR TOOLS:**
You have access to these tools:
- calculator: For mathematical calculations
- get_current_time: For current time information  
- knowledge_base_search: For general factual information
- search_zus_website: For general ZUS Coffee information
- search_zus_products: For ZUS Coffee drinkware and product queries
- search_zus_outlets: For ZUS Coffee store locations and outlet information

**DECISION PROCESS:**
1. **Parse Intent**: Analyze what the user is asking for
2. **Identify Missing Information**: Check if you have all necessary information
3. **Choose Action**: Decide which tool to use or ask for clarification

**PLANNING GUIDELINES:**
- For ZUS Coffee products/drinkware questions → Use search_zus_products
- For ZUS Coffee outlets/stores/locations → Use search_zus_outlets  
- For general ZUS Coffee questions → Use search_zus_website
- For calculations → Use calculator
- For time → Use get_current_time
- For general knowledge → Use knowledge_base_search

**EXAMPLES:**
User: "What ZUS Coffee drinkware do you have?" → Use search_zus_products
User: "Where are ZUS Coffee outlets in KL?" → Use search_zus_outlets
User: "What time is it?" → Use get_current_time
User: "Calculate 15 * 23" → Use calculator

Always be helpful and use the appropriate tool for each query.
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        # Create agent executor with memory
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,  # Memory is added here
            verbose=settings.DEBUG,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def plan_and_execute(self, user_input: str, chat_history: List = None) -> Dict[str, Any]:
        """
        Main planning and execution method with memory support
        """
        try:
            logger.info(f"Processing user input: {user_input}")
            
            # Parse intent
            intent_analysis = self._analyze_intent(user_input)
            logger.info(f"Intent Analysis: {intent_analysis}")
            
            # Execute with agent (memory is automatically handled)
            result = self.agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history or []
            })
            
            # Extract decision information
            intermediate_steps = result.get("intermediate_steps", [])
            tools_used = self._extract_tools_used(intermediate_steps)
            decision_points = self._extract_decision_points(intermediate_steps)

            planning_info = {
                "user_input": user_input,
                "intent_analysis": intent_analysis,
                "final_answer": result["output"],
                "tools_used": tools_used,
                "decision_points": decision_points,
                "success": True,
                "memory_used": bool(self.memory.chat_memory.messages)
            }
            
            logger.info(f"Tools used: {tools_used}")
            return planning_info
            
        except Exception as e:
            logger.error(f"Error in plan_and_execute: {e}")
            return {
                "user_input": user_input,
                "intent_analysis": "Error in analysis",
                "final_answer": f"I encountered an error: {str(e)}. Please try again.",
                "tools_used": [],
                "decision_points": [f"Error: {str(e)}"],
                "success": False,
                "memory_used": False
            }
    
    def _analyze_intent(self, user_input: str) -> str:
        """Analyze user intent for planning purposes"""
        user_input_lower = user_input.lower()
        
        # ZUS Coffee specific intents
        if any(word in user_input_lower for word in ['zus', 'coffee']):
            if any(word in user_input_lower for word in ['product', 'drinkware', 'tumbler', 'mug', 'cup', 'bottle']):
                return "ZUS Coffee product search required"
            elif any(word in user_input_lower for word in ['outlet', 'store', 'location', 'branch', 'address', 'where']):
                return "ZUS Coffee outlet search required"
            else:
                return "General ZUS Coffee information required"
        
        # General intents
        elif any(word in user_input_lower for word in ['calculate', 'math', '+', '-', '*', '/', 'equals']):
            return "Mathematical calculation required"
        elif any(word in user_input_lower for word in ['time', 'date', 'now', 'current']):
            return "Time information required"
        else:
            return "General conversation or knowledge retrieval"
    
    def _extract_tools_used(self, intermediate_steps: List) -> List[str]:
        """Extract which tools were used during execution"""
        tools_used = []
        for step in intermediate_steps:
            if len(step) >= 1 and hasattr(step[0], 'tool'):
                tools_used.append(step[0].tool)
        return tools_used
    
    def _extract_decision_points(self, intermediate_steps: List) -> List[str]:
        """Extract decision points from the agent's reasoning"""
        decision_points = []
        for i, step in enumerate(intermediate_steps):
            if len(step) >= 2:
                action = step[0]
                observation = step[1]
                decision_points.append(
                    f"Decision {i+1}: Used {action.tool} → {str(observation)[:100]}..."
                )
        return decision_points