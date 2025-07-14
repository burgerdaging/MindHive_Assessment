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
        """Search ZUS products using vector store and live search only"""
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
                    logger.info(f"Found {len(vector_results)} results in vector store")
                else:
                    logger.info("No results found in vector store")
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                results["vector_results"] = []
            
            # # 2. Search live website
            # try:
            #     live_results = self._search_live_website(query)
            #     results["live_search_results"] = live_results
            #     if live_results:
            #         results["sources"].append("Live Website")
            #         logger.info(f"Found {len(live_results)} results from live search")
            #     else:
            #         logger.info("No results found from live search")
            # except Exception as e:
            #     logger.warning(f"Live search failed: {e}")
            #     results["live_search_results"] = []
            
            # 3. Generate combined summary
            results["combined_summary"] = await self._generate_combined_summary(query, results)
            
            # 4. Check if we found anything at all
            total_results = len(results["vector_results"]) + len(results["live_search_results"])
            if total_results == 0:
                results["success"] = False
                results["combined_summary"] = self._generate_no_results_message(query)
            
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
            # Enhanced query for better product search
            enhanced_query = f"{query} ZUS Coffee drinkware products tumbler mug cup bottle"
            
            vector_search = MongoDBAtlasVectorSearch(
                embedding=self.embeddings,
                collection=self.collection,
                index_name=self.vector_index_name,
            )
            
            retriever = vector_search.as_retriever(search_kwargs={"k": top_k})
            relevant_docs = retriever.get_relevant_documents(enhanced_query)
            
            # Filter for product-related content
            product_results = []
            for doc in relevant_docs:
                content_lower = doc.page_content.lower()
                # Check if document contains product-related keywords
                if any(word in content_lower for word in [
                    "drinkware", "tumbler", "mug", "cup", "bottle", "flask", 
                    "product", "price", "rm", "ringgit", "buy", "shop", "purchase"
                ]):
                    product_results.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "ZUS Document"),
                        "type": "stored_document",
                        "metadata": doc.metadata
                    })
            
            logger.info(f"Vector search found {len(product_results)} product-related documents out of {len(relevant_docs)} total documents")
            return product_results
            
        except Exception as e:
            logger.error(f"Vector store search error: {e}")
            return []
    
    def _search_live_website(self, query: str) -> List[Dict]:
        """Search live ZUS website for products"""
        try:
            # Multiple search strategies for better coverage
            search_queries = [
                f"{query} drinkware ZUS Coffee shop",
                f"ZUS Coffee {query} products",
                f"{query} tumbler mug ZUS Coffee"
            ]
            
            all_results = []
            
            for search_query in search_queries:
                try:
                    search_results = self.search_service.search_zus_website(search_query, num_results=3)
                    
                    if search_results["success"] and search_results["results"]:
                        for result in search_results["results"]:
                            # Filter for product-related results
                            if self._is_product_related(result):
                                result_data = {
                                    "title": result["title"],
                                    "url": result["link"],
                                    "description": result["snippet"],
                                    "category": result["category"],
                                    "type": "live_search",
                                    "search_query": search_query
                                }
                                
                                # Avoid duplicates
                                if not any(existing["url"] == result_data["url"] for existing in all_results):
                                    all_results.append(result_data)
                
                except Exception as e:
                    logger.warning(f"Search query '{search_query}' failed: {e}")
                    continue
            
            logger.info(f"Live search found {len(all_results)} unique product-related results")
            return all_results[:5]  # Limit to top 5 results
            
        except Exception as e:
            logger.error(f"Live search error: {e}")
            return []
    
    def _is_product_related(self, result: Dict) -> bool:
        """Check if search result is product-related"""
        text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
        
        product_indicators = [
            "drinkware", "tumbler", "mug", "cup", "bottle", "flask",
            "shop", "buy", "purchase", "product", "price", "rm",
            "collection", "merchandise", "accessories"
        ]
        
        return any(indicator in text for indicator in product_indicators)
    
    async def _generate_combined_summary(self, query: str, results: Dict) -> str:
        """Generate AI summary combining all sources"""
        try:
            context_parts = []
            
            # Add vector store results
            if results["vector_results"]:
                context_parts.append("=== ZUS Product Documentation ===")
                for result in results["vector_results"]:
                    context_parts.append(result["content"])
            
            # # Add live search results
            # if results["live_search_results"]:
            #     context_parts.append("\n=== Live Website Information ===")
            #     for result in results["live_search_results"]:
            #         context_parts.append(f"Title: {result['title']}\nDescription: {result['description']}\nURL: {result['url']}")
            
            if not context_parts:
                return self._generate_no_results_message(query)
            
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
            6. If live website links are available, mention them for current information
            7. If information is limited, acknowledge this and suggest where to find more details

            Summary:
            """
            
            response = self.llm.invoke(summary_prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"I found some product information but couldn't generate a proper summary. Error: {str(e)}"
    
    def _generate_no_results_message(self, query: str) -> str:
        """Generate helpful message when no results are found"""
        return f"""I couldn't find specific information about "{query}" in our current ZUS Coffee product database or website search.

This could be because:
1. The product information hasn't been indexed in our vector store yet
2. The search terms don't match available product descriptions
3. The product might not be currently available

**Suggestions:**
- Try different search terms like "tumbler", "mug", "drinkware", or "coffee cup"
- Visit the official ZUS Coffee website directly at shop.zuscoffee.com
- Check ZUS Coffee's physical stores for the latest product catalog
- Contact ZUS Coffee customer service for specific product inquiries

**Popular ZUS Coffee drinkware categories to search for:**
- Tumblers
- Travel mugs
- Coffee cups
- Water bottles
- Thermal flasks

Would you like me to search for any of these specific categories?"""