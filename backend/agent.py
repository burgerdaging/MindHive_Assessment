import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
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
from typing import List, Dict, Any, Optional
import json
import datetime
import requests
import sqlite3
import re
from urllib.parse import quote_plus
import asyncio

#Part 2 & 3 & 4 & 5 endpoints are here
load_dotenv()
app = FastAPI(title="ZUS Coffee AI Assistant", description="AI Assistant with Agentic Planning and ZUS Coffee Integration")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY, temperature=0.0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
client = MongoClient(os.getenv("MONGODB_ATLAS_CLUSTER_URI"))

DB_NAME = "MongoDB"
COLLECTION_NAME = "Facts-txt"
COLLECTION_NAME_TWO = "Zus-Coffee-Document.pdf"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "MindHive-Assessment"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

#Zus Product Collection
ZUS_PRODUCTS_COLLECTION = client[DB_NAME][COLLECTION_NAME_TWO]
ZUS_VECTOR_INDEX_NAME = "MindHive-Assessment"

# ============= ENHANCED PYDANTIC MODELS =============

class QueryRequest(BaseModel):
    message: str
    chat_history: List[Dict] = []

class QueryResponse(BaseModel):
    response: str
    tools_used: List[str] = []
    decision_points: List[str] = []
    success: bool = True

class ProductQueryResponse(BaseModel):
    summary: str
    products: List[Dict[str, Any]]
    query: str
    total_found: int
    success: bool = True
    sources: List[str] = []

class OutletQueryResponse(BaseModel):
    summary: str
    outlets: List[Dict[str, Any]]
    query: str
    total_found: int
    success: bool = True
    sources: List[str] = []

# ============= GOOGLE CUSTOM SEARCH ENGINE INTEGRATION =============

class GoogleCustomSearchTool:
    def __init__(self):
        self.api_key = GOOGLE_SEARCH_API_KEY
        self.search_engine_id = SEARCH_ENGINE_ID
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    def search_zus_website(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """Search ZUS Coffee website using Google Custom Search"""
        try:
            print(f"🔍 Searching ZUS website for: {query}")
            
            if not self.api_key:
                return {
                    "query": query,
                    "results": [],
                    "error": "Google Search API key not configured",
                    "success": False
                }
            
            # Prepare search parameters
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10),
                'safe': 'active'
            }
            
            # Make API request with error handling
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            search_data = response.json()
            processed_results = self._process_search_results(search_data, query)
            
            return {
                "query": query,
                "results": processed_results,
                "total_results": search_data.get("searchInformation", {}).get("totalResults", "0"),
                "search_time": search_data.get("searchInformation", {}).get("searchTime", "0"),
                "success": True
            }
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Search API error: {e}")
            return {
                "query": query,
                "results": [],
                "error": f"Search API error: {str(e)}",
                "success": False
            }
        except Exception as e:
            print(f"❌ Unexpected search error: {e}")
            return {
                "query": query,
                "results": [],
                "error": f"Unexpected error: {str(e)}",
                "success": False
            }
    
    def _process_search_results(self, search_data: Dict, query: str) -> List[Dict]:
        """Process and clean search results"""
        processed_results = []
        items = search_data.get("items", [])
        
        for item in items:
            try:
                result = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "display_link": item.get("displayLink", ""),
                    "category": self._detect_category(item)
                }
                result["snippet"] = self._clean_snippet(result["snippet"])
                processed_results.append(result)
            except Exception as e:
                print(f"❌ Error processing search result: {e}")
                continue
        
        return processed_results
    
    def _detect_category(self, item: Dict) -> str:
        """Detect if result is about products, outlets, or general info"""
        title = item.get("title", "").lower()
        link = item.get("link", "").lower()
        snippet = item.get("snippet", "").lower()
        combined_text = f"{title} {link} {snippet}"
        
        if any(word in combined_text for word in ["drinkware", "tumbler", "mug", "cup", "bottle", "flask", "shop"]):
            return "product"
        elif any(word in combined_text for word in ["store", "outlet", "location", "address", "branch"]):
            return "outlet"
        else:
            return "general"
    
    def _clean_snippet(self, snippet: str) -> str:
        """Clean and format snippet text"""
        if not snippet:
            return ""
        cleaned = re.sub(r'\s+', ' ', snippet.strip())
        cleaned = re.sub(r'^\d+\s*[.-]\s*', '', cleaned)
        return cleaned

# Initialize search tool
google_search_tool = GoogleCustomSearchTool()

# ============= ZUS PRODUCTS SERVICE (Vector Store Only) =============

class ZUSProductsService:
    def __init__(self):
        self.embeddings = embeddings
        self.collection = ZUS_PRODUCTS_COLLECTION
        self.llm = llm
        self.vector_index_name = ZUS_VECTOR_INDEX_NAME
        self.search_tool = google_search_tool
    
    async def search_products(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search ZUS products using vector store and live search"""
        try:
            print(f"🔍 Searching ZUS products for: {query}")
            
            results = {
                "query": query,
                "vector_results": [],
                "live_search_results": [],
                "combined_summary": "",
                "success": True,
                "sources": []
            }
            
            # 1. Search vector store (your ZUS document)
            try:
                vector_results = await self._search_vector_store(query, top_k)
                results["vector_results"] = vector_results
                if vector_results:
                    results["sources"].append("ZUS Document Store")
            except Exception as e:
                print(f"⚠️ Vector search failed: {e}")
                results["vector_results"] = []
            
            # 2. Search live website as backup
            try:
                live_results = self._search_live_website(query)
                results["live_search_results"] = live_results
                if live_results:
                    results["sources"].append("Live Website")
            except Exception as e:
                print(f"⚠️ Live search failed: {e}")
                results["live_search_results"] = []
            
            # 3. Generate combined summary
            results["combined_summary"] = await self._generate_combined_summary(query, results)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in product search: {e}")
            return {
                "query": query,
                "vector_results": [],
                "live_search_results": [],
                "combined_summary": f"Search error: {str(e)}",
                "success": False,
                "sources": []
            }
    
    async def _search_vector_store(self, query: str, top_k: int) -> List[Dict]:
        """Search ZUS products in vector store"""
        try:
            # Enhanced query for better product search
            enhanced_query = f"{query} ZUS Coffee drinkware products tumbler mug"
            
            vector_search = MongoDBAtlasVectorSearch(
                embedding=self.embeddings,
                collection=self.collection,
                index_name=self.vector_index_name,
            )
            
            retriever = vector_search.as_retriever(search_kwargs={"k": top_k})
            relevant_docs = retriever.get_relevant_documents(enhanced_query)
            
            return [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "ZUS Document"),
                    "type": "stored_document"
                }
                for doc in relevant_docs
            ]
            
        except Exception as e:
            print(f"❌ Vector store search error: {e}")
            return []
    
    def _search_live_website(self, query: str) -> List[Dict]:
        """Search live ZUS website for products"""
        try:
            enhanced_query = f"{query} drinkware products ZUS Coffee shop"
            search_results = self.search_tool.search_zus_website(enhanced_query, num_results=5)
            
            if not search_results["success"]:
                return []
            
            product_results = []
            for result in search_results["results"]:
                if result["category"] in ["product", "general"]:
                    product_results.append({
                        "title": result["title"],
                        "url": result["link"],
                        "description": result["snippet"],
                        "category": result["category"],
                        "type": "live_search"
                    })
            
            return product_results
            
        except Exception as e:
            print(f"❌ Live search error: {e}")
            return []
    
    async def _generate_combined_summary(self, query: str, results: Dict) -> str:
        """Generate AI summary combining all sources"""
        try:
            context_parts = []
            
            if results["vector_results"]:
                context_parts.append("=== ZUS Product Documentation ===")
                for result in results["vector_results"]:
                    context_parts.append(result["content"])
            
            if results["live_search_results"]:
                context_parts.append("\n=== Live Website Information ===")
                for result in results["live_search_results"]:
                    context_parts.append(f"Title: {result['title']}\nDescription: {result['description']}\nURL: {result['url']}")
            
            if not context_parts:
                return "No relevant ZUS Coffee drinkware products found for your query."
            
            combined_context = "\n\n".join(context_parts)
            
            summary_prompt = f"""
            Based on the following information about ZUS Coffee drinkware products, provide a comprehensive and helpful summary for the user's query: "{query}"

            Available Information:
            {combined_context}

            Instructions:
            1. Provide a clear, helpful summary that directly addresses the user's query
            2. Highlight relevant drinkware products and their features
            3. Include specific details like prices, materials, and features when available
            4. If you have both stored and live information, combine them intelligently
            5. Keep the response customer-focused and actionable
            6. If live website links are available, mention them

            Summary:
            """
            
            response = self.llm.invoke(summary_prompt)
            return response.content.strip()
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
            return f"Unable to generate summary: {str(e)}"



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

