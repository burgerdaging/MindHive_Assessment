import requests
import re
from typing import Dict, List, Any
from config import settings
import logging

logger = logging.getLogger(__name__)

class GoogleCustomSearchService:
    def __init__(self):
        self.api_key = settings.GOOGLE_SEARCH_API_KEY
        self.search_engine_id = settings.SEARCH_ENGINE_ID
        self.base_url = "https://cse.google.com/cse?cx=b6fecb64edd314067"
        
    def search_zus_website(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """Search ZUS Coffee website using Google Custom Search"""
        try:
            logger.info(f"Searching ZUS website for: {query}")
            
            if not self.api_key or not self.search_engine_id:
                return {
                    "query": query,
                    "results": [],
                    "error": "Google Search API not configured",
                    "success": False
                }
            
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10),
                'safe': 'active'
            }
            
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
            logger.error(f"Search API error: {e}")
            return {
                "query": query,
                "results": [],
                "error": f"Search API error: {str(e)}",
                "success": False
            }
        except Exception as e:
            logger.error(f"Unexpected search error: {e}")
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
                logger.error(f"Error processing search result: {e}")
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