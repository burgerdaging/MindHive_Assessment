from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from services.search_service import GoogleCustomSearchService
from config import settings
import logging

logger = logging.getLogger(__name__)

class ZUSProductsService:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=settings.GEMINI_API_KEY
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            api_key=settings.GEMINI_API_KEY, 
            temperature=0.0
        )
        
        # MongoDB setup
        self.client = MongoClient(settings.MONGODB_ATLAS_CLUSTER_URI)
        self.collection = self.client[settings.DB_NAME][settings.ZUS_COLLECTION_NAME]
        self.vector_index_name = settings.ATLAS_VECTOR_SEARCH_INDEX_NAME
        
        # Search service
        self.search_service = GoogleCustomSearchService()
    
    async def search_products(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search ZUS products using vector store and live search"""
        try:
            logger.info(f"Searching ZUS products for: {query}")
            
            results = {
                "query": query,
                "vector_results": [],
                "live_search_results": [],
                "combined_summary": "",
                "success": True,
                "sources": []
            }
            
            # 1. Search vector store
            try:
                vector_results = await self._search_vector_store(query, top_k)
                results["vector_results"] = vector_results
                if vector_results:
                    results["sources"].append("ZUS Document Store")
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                results["vector_results"] = []
            
            # 2. Search live website as backup
            try:
                live_results = self._search_live_website(query)
                results["live_search_results"] = live_results
                if live_results:
                    results["sources"].append("Live Website")
            except Exception as e:
                logger.warning(f"Live search failed: {e}")
                results["live_search_results"] = []
            
            # 3. Generate combined summary
            results["combined_summary"] = await self._generate_combined_summary(query, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in product search: {e}")
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
            logger.error(f"Vector store search error: {e}")
            return []
    
    def _search_live_website(self, query: str) -> List[Dict]:
        """Search live ZUS website for products"""
        try:
            enhanced_query = f"{query} drinkware products ZUS Coffee shop"
            search_results = self.search_service.search_zus_website(enhanced_query, num_results=5)
            
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
            logger.error(f"Live search error: {e}")
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
            logger.error(f"Error generating summary: {e}")
            return f"Unable to generate summary: {str(e)}"