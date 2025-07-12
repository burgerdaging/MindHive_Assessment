import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.agents import tool, AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from pymongo import MongoClient 
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Any
import json
import datetime

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY, temperature=0.0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))

DB_NAME = "MongoDB"
COLLECTION_NAME = "Facts-txt"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "MindHive-Assessment"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# ============= TOOLS IMPLEMENTATION =============

@tool
def get_current_time(query: str) -> str:
    """Returns the current time. Use this tool when the user asks for the current time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression safely. Use this for any math questions.
    Supports basic arithmetic: +, -, *, /, (), and basic functions like sqrt, sin, cos.
    """
    try:
        # Basic safety check - only allow safe characters
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression.replace(' ', '')):
            return "Error: Invalid characters in expression. Only numbers and basic operators allowed."
        
        # Evaluate safely
        result = eval(expression)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

@tool
def knowledge_base_search(query: str) -> str:
    """
    Searches the MongoDB Atlas vector database for relevant information.
    Use this tool when users ask factual questions or need information retrieval.
    """
    try:
        # Create the vector search instance
        vector_search = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=MONGODB_COLLECTION,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        )
        
        # Create retrieval chain
        retrieval_qa_chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the following question based on the provided context:\n\n{context}"),
            ("human", "{input}"),
        ])
        
        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        retrieval_chain = create_retrieval_chain(
            retriever=vector_search.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain=combine_docs_chain
        )
        
        # Execute the search
        result = retrieval_chain.invoke({"input": query})
        return result["answer"]
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

# ============= AGENTIC PLANNING SYSTEM =============

tools = [get_current_time, calculator, knowledge_base_search]

# Enhanced prompt for better agentic planning
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent AI assistant with advanced planning capabilities. 

**DECISION PROCESS:**
1. **Parse Intent**: Analyze what the user is asking for
2. **Identify Missing Information**: Check if you have all necessary information
3. **Choose Action**: Decide whether to:
   - Ask a clarifying question if information is missing
   - Use calculator tool for mathematical operations
   - Use knowledge_base_search for factual information
   - Use get_current_time for time-related queries
   - Provide direct answer if no tools needed

**PLANNING GUIDELINES:**
- If user asks for calculation but doesn't provide numbers, ASK for clarification
- If user asks factual questions, use knowledge_base_search
- If user asks for time, use get_current_time
- Always explain your reasoning process
- Handle errors gracefully

**EXAMPLES:**
User: "Calculate something" → Ask: "What would you like me to calculate? Please provide the mathematical expression."
User: "What is 15 * 23?" → Use calculator tool
User: "Tell me about Python" → Use knowledge_base_search tool
User: "What time is it?" → Use get_current_time tool
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent with enhanced planning
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# ============= PYDANTIC MODELS =============

class QueryRequest(BaseModel):
    message: str
    chat_history: List[Dict] = []

class QueryResponse(BaseModel):
    response: str
    tools_used: List[str] = []
    decision_points: List[str] = []
    success: bool = True

# ============= AGENTIC PLANNER CLASS =============

class AgenticPlanner:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def plan_and_execute(self, user_input: str, chat_history: List = None) -> Dict[str, Any]:
        """
        Main planning and execution method that demonstrates the decision-making process
        """
        try:
            print(f"\n🤖 PLANNING: Processing '{user_input}'")
            
            # Step 1: Parse Intent and Missing Information
            intent_analysis = self._analyze_intent(user_input)
            print(f"📊 Intent Analysis: {intent_analysis}")
            
            # Step 2: Execute with agent
            if chat_history is None:
                chat_history = []

            #TODO DISPLAY TOOLS AND OTHERS TO SHOW IT USES THE TOOLS GIVEN
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=5,
                handle_parsing_errors=True,
                return_intermediate_steps=True  # This is the key fix!
            )

            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })
            print(f"🔍 Debug - Result keys: {result.keys()}")
            print(f"🔍 Debug - Intermediate steps: {result.get('intermediate_steps', 'Not found')}")

            # Step 3: Extract decision information
            intermediate_steps = result.get("intermediate_steps", [])
            tools_used = self._extract_tools_used(intermediate_steps)
            decision_points = self._extract_decision_points(intermediate_steps)

            planning_info = {
                "user_input": user_input,
                "intent_analysis": intent_analysis,
                "final_answer": result["output"],
                "tools_used": self._extract_tools_used(result.get("intermediate_steps", [])),
                "decision_points": self._extract_decision_points(result.get("intermediate_steps", [])),
                "success": True,
                "debug_info": {
                    "intermediate_steps_count": len(intermediate_steps),
                    "result_keys": list(result.keys())
                }
            }
            print(f"🔧 Debug - Tools used: {tools_used}")
            print(f"🔧 Debug - Decision points: {decision_points}")
            return planning_info
            
        except Exception as e:
            return {
                "user_input": user_input,
                "intent_analysis": "Error in analysis",
                "final_answer": f"I encountered an error: {str(e)}",
                "tools_used": [],
                "decision_points": [f"Error occurred: {str(e)}"],
                "success": False
            }
    
    def _analyze_intent(self, user_input: str) -> str:
        """Analyze user intent for planning purposes"""
        user_input_lower = user_input.lower()
        
        if any(word in user_input_lower for word in ['calculate', 'math', '+', '-', '*', '/', 'equals']):
            return "Mathematical calculation required"
        elif any(word in user_input_lower for word in ['time', 'date', 'now', 'current']):
            return "Time information required"
        elif any(word in user_input_lower for word in ['what', 'who', 'where', 'when', 'why', 'how', 'tell me', 'explain']):
            return "Knowledge retrieval required"
        else:
            return "General conversation"
    
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
                    f"Decision {i+1}: Used {action.tool} → {observation[:100]}..."
                )
        return decision_points

# Initialize planner
planner = AgenticPlanner()

# ============= FASTAPI ENDPOINTS =============

@app.post("/agent", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Main chat endpoint that demonstrates agentic planning and tool calling
    """
    try:
        result = planner.plan_and_execute(request.message, request.chat_history)
        
        return QueryResponse(
            response=result["final_answer"],
            tools_used=result["tools_used"],
            decision_points=result["decision_points"],
            success=result["success"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test_endpoint():
    """Test endpoint to verify the system is working"""
    return {"message": "Agentic Planning System is running!", "tools": [tool.name for tool in tools]}

# ============= TEST SCENARIOS =============

def run_test_scenarios():
    """Run test scenarios to demonstrate the system"""
    test_cases = [
        "Hello, how are you?",  # No tool needed
        "What time is it?",     # Time tool
        "Calculate 15 * 23",    # Calculator tool
        "What is Python?",      # Knowledge base search
        "Calculate something",  # Missing information - should ask for clarification
        "10 / 0",              # Error handling
    ]
    
    print("\n" + "="*60)
    print("🧪 RUNNING TEST SCENARIOS")
    print("="*60)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_input} ---")
        result = planner.plan_and_execute(test_input)
        print(f"Response: {result['final_answer']}")
        print(f"Tools Used: {result['tools_used']}")
        print(f"Decision Points: {result['decision_points']}")

