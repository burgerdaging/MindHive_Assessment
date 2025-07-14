from langchain.tools import tool
from services.products_service import ZUSProductsService
from services.outlets_service import ZUSOutletsService
from services.search_service import GoogleCustomSearchService
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pymongo import MongoClient
from config import settings
import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)

# Initialize services
products_service = ZUSProductsService()
outlets_service = ZUSOutletsService()
search_service = GoogleCustomSearchService()

# Initialize LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=settings.GEMINI_API_KEY, temperature=0.0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=settings.GEMINI_API_KEY)
client = MongoClient(settings.MONGODB_ATLAS_CLUSTER_URI)
mongodb_collection = client[settings.DB_NAME][settings.COLLECTION_NAME]

def run_async_in_sync(coro):
    """Helper function to run async code in sync context"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to use a different approach
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        else:
            # If no loop is running, we can run directly
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

@tool
def get_current_time(query: str) -> str:
    """Returns the current time. Use this tool when the user asks for the current time."""
    try:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.error(f"Error getting current time: {e}")
        return f"Error getting current time: {str(e)}"

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression safely. Use this for any math questions.
    Supports basic arithmetic: +, -, *, /, (), and basic functions like sqrt, sin, cos.
    """
    try:
        # Input validation for security
        if not expression or not expression.strip():
            return "Error: Empty expression provided. Please provide a mathematical expression to calculate."
        
        # Basic safety check - only allow safe characters
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression.replace(' ', '')):
            return "Error: Invalid characters in expression. Only numbers and basic operators (+, -, *, /, parentheses) are allowed."
        
        # Additional security: check for dangerous patterns
        dangerous_patterns = ['import', 'exec', 'eval', '__', 'open', 'file']
        if any(pattern in expression.lower() for pattern in dangerous_patterns):
            return "Error: Potentially dangerous expression detected. Please use only mathematical operations."
        
        # Evaluate safely
        result = eval(expression)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except SyntaxError:
        return "Error: Invalid mathematical expression. Please check your syntax."
    except Exception as e:
        logger.error(f"Calculator error: {e}")
        return f"Error evaluating expression: {str(e)}"

@tool
def knowledge_base_search(query: str) -> str:
    """
    Searches the MongoDB Atlas vector database for relevant information.
    Use this tool when users ask factual questions or need information retrieval.
    """
    try:
        # Input validation
        if not query or not query.strip():
            return "Error: Empty search query provided. Please provide a specific question or topic to search for."
        
        # Create the vector search instance
        vector_search = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=mongodb_collection,
            index_name=settings.ATLAS_VECTOR_SEARCH_INDEX_NAME,
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
        logger.error(f"Knowledge base search error: {e}")
        return f"Error searching knowledge base: {str(e)}. Please try rephrasing your question."

@tool
def search_zus_website(query: str) -> str:
    """
    Search ZUS Coffee website for products, outlets, or general information.
    Use this tool when users ask about ZUS Coffee products, store locations, or general company information.
    """
    try:
        # Input validation
        if not query or not query.strip():
            return "Error: Empty search query provided. Please specify what you'd like to search for on the ZUS Coffee website."
        
        # Perform search with error handling
        search_results = search_service.search_zus_website(query.strip())
        
        if not search_results["success"]:
            error_msg = search_results.get('error', 'Unknown error')
            return f"Search failed: {error_msg}. Please try again later or rephrase your query."
        
        if not search_results["results"]:
            return f"No results found for '{query}' on ZUS Coffee website. Try using different keywords or check the spelling."
        
        # Format results for the agent
        formatted_results = []
        for i, result in enumerate(search_results["results"][:5], 1):
            formatted_result = f"""
{i}. {result['title']}
   URL: {result['link']}
   Info: {result['snippet']}
   Category: {result['category']}
"""
            formatted_results.append(formatted_result)
        
        response = f"""Found {len(search_results['results'])} results for '{query}':

{''.join(formatted_results)}

Total results available: {search_results['total_results']}
Search completed in: {search_results['search_time']} seconds"""
        
        return response
        
    except Exception as e:
        logger.error(f"Website search error: {e}")
        return f"Error performing search: {str(e)}. Please try again later."

@tool
def search_zus_products(query: str) -> str:
    """
    Search ZUS Coffee drinkware products using vector store and live website.
    Use this tool when users ask specifically about ZUS Coffee products, drinkware, tumblers, mugs, etc.
    """
    try:
        # Input validation
        if not query or not query.strip():
            return "Error: Empty product search query. Please specify what ZUS Coffee products you're looking for (e.g., 'tumblers', 'coffee mugs', 'drinkware')."
        
        # Use the helper function to run async code
        result = run_async_in_sync(products_service.search_products(query.strip()))
        
        if not result["success"]:
            return f"Product search failed: {result.get('combined_summary', 'Unknown error')}. Please try again with different keywords."
        
        return result["combined_summary"]
        
    except Exception as e:
        logger.error(f"Product search error: {e}")
        return f"Error searching ZUS products: {str(e)}. Please try again later or contact support."

@tool
def search_zus_outlets(query: str) -> str:
    """
    Search ZUS Coffee outlet locations using vector store and live website.
    Use this tool when users ask about ZUS Coffee store locations, outlets, branches, addresses, or operating hours.
    """
    try:
        # Input validation
        if not query or not query.strip():
            return "Error: Empty outlet search query. Please specify what you're looking for (e.g., 'outlets in KL', 'store hours', 'nearest branch')."
        
        # Use the helper function to run async code
        result = run_async_in_sync(outlets_service.search_outlets(query.strip()))
        
        if not result["success"]:
            return f"Outlet search failed: {result.get('combined_summary', 'Unknown error')}. Please try again with different keywords."
        
        return result["combined_summary"]
        
    except Exception as e:
        logger.error(f"Outlet search error: {e}")
        return f"Error searching ZUS outlets: {str(e)}. Please try again later or contact support."

# Export all tools
AVAILABLE_TOOLS = [
    get_current_time,
    calculator,
    knowledge_base_search,
    search_zus_website,
    search_zus_products,
    search_zus_outlets
]