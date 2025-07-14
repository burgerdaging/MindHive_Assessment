import requests
import re
from typing import Dict, List, Any
from config import settings
import logging
import json

logger = logging.getLogger(__name__)

class GoogleCustomSearchService:
    def __init__(self):
        self.api_key = settings.GOOGLE_SEARCH_API_KEY
        self.search_engine_id = settings.SEARCH_ENGINE_ID
        self.base_url = settings.SEARCH_PUBLIC_URL
        
        # Log configuration status
        self._log_configuration()
        
    def _log_configuration(self):
        """Log the configuration status for debugging"""
        logger.info("Google Custom Search Configuration:")
        logger.info(f"API Key configured: {'Yes' if self.api_key else 'No'}")
        logger.info(f"Search Engine ID configured: {'Yes' if self.search_engine_id else 'No'}")
        
        if self.api_key:
            logger.info(f"API Key preview: {self.api_key[:10]}...")
        if self.search_engine_id:
            logger.info(f"Search Engine ID: {self.search_engine_id}")
        
    def search_zus_website(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """Search ZUS Coffee website using Google Custom Search"""
        try:
            logger.info(f"Searching ZUS website for: {query}")
            
            # Check if API is configured
            if not self.api_key or not self.search_engine_id:
                logger.warning("Google Search API not properly configured")
                return {
                    "query": query,
                    "results": [],
                    "error": "Google Search API not configured. Please set GOOGLE_SEARCH_API_KEY and SEARCH_ENGINE_ID in your .env file",
                    "success": False,
                    "fallback_used": True
                }
            
            # Prepare search parameters
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10),
                'safe': 'active'
            }
            
            logger.info(f"Making request to: {self.base_url}")
            logger.info(f"Parameters: {dict(params, key='[HIDDEN]')}")  # Hide API key in logs
            
            # Make API request with detailed error handling
            response = requests.get(self.base_url, params=params, timeout=10)
            
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            # Check if response is successful
            response.raise_for_status()
            
            # Check if response has content
            if not response.text:
                logger.error("Empty response from Google Search API")
                return self._create_fallback_response(query, "Empty response from Google Search API")
            
            # Try to parse JSON
            try:
                search_data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                logger.error(f"Response content: {response.text[:500]}...")
                return self._create_fallback_response(query, f"Invalid JSON response: {str(e)}")
            
            # Check for API errors in response
            if 'error' in search_data:
                error_info = search_data['error']
                logger.error(f"Google API error: {error_info}")
                return self._create_fallback_response(query, f"Google API error: {error_info.get('message', 'Unknown error')}")
            
            # Process search results
            processed_results = self._process_search_results(search_data, query)
            
            return {
                "query": query,
                "results": processed_results,
                "total_results": search_data.get("searchInformation", {}).get("totalResults", "0"),
                "search_time": search_data.get("searchInformation", {}).get("searchTime", "0"),
                "success": True,
                "fallback_used": False
            }
            
        except requests.exceptions.Timeout:
            logger.error("Google Search API timeout")
            return self._create_fallback_response(query, "Search API timeout")
            
        except requests.exceptions.ConnectionError:
            logger.error("Connection error to Google Search API")
            return self._create_fallback_response(query, "Connection error to search API")
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from Google Search API: {e}")
            logger.error(f"Response content: {e.response.text if e.response else 'No response'}")
            
            # Handle specific HTTP errors
            if e.response and e.response.status_code == 403:
                return self._create_fallback_response(query, "API quota exceeded or access denied. Please check your Google Search API configuration.")
            elif e.response and e.response.status_code == 400:
                return self._create_fallback_response(query, "Invalid request parameters. Please check your Search Engine ID.")
            else:
                return self._create_fallback_response(query, f"HTTP error: {str(e)}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return self._create_fallback_response(query, f"Request error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Unexpected error in search: {e}")
            return self._create_fallback_response(query, f"Unexpected error: {str(e)}")
    
    def _create_fallback_response(self, query: str, error_message: str) -> Dict[str, Any]:
        """Create a fallback response when search fails"""
        # Create some basic fallback results for ZUS Coffee
        fallback_results = []
        
        if any(word in query.lower() for word in ['product', 'drinkware', 'tumbler', 'mug', 'cup']):
            fallback_results = [
                {
                    "title": "ZUS Coffee Drinkware Collection",
                    "link": "https://shop.zuscoffee.com/collections/drinkware",
                    "snippet": "Explore ZUS Coffee's collection of premium drinkware including tumblers, mugs, and travel cups.",
                    "display_link": "shop.zuscoffee.com",
                    "category": "product"
                }
            ]
        elif any(word in query.lower() for word in ['outlet', 'store', 'location', 'branch']):
            fallback_results = [
                {
                    "title": "ZUS Coffee Store Locations",
                    "link": "https://zuscoffee.com/store-locator",
                    "snippet": "Find ZUS Coffee outlets near you. Locations in Kuala Lumpur, Selangor, and other major cities.",
                    "display_link": "zuscoffee.com",
                    "category": "outlet"
                }
            ]
        
        return {
            "query": query,
            "results": fallback_results,
            "total_results": str(len(fallback_results)),
            "search_time": "0",
            "success": False,
            "error": error_message,
            "fallback_used": True
        }
    
    def _process_search_results(self, search_data: Dict, query: str) -> List[Dict]:
        """Process and clean search results"""
        processed_results = []
        items = search_data.get("items", [])
        
        logger.info(f"Processing {len(items)} search results")
        
        for i, item in enumerate(items):
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
                
                logger.debug(f"Processed result {i+1}: {result['title']}")
                
            except Exception as e:
                logger.error(f"Error processing search result {i+1}: {e}")
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
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the Google Search API connection"""
        try:
            logger.info("Testing Google Search API connection...")
            result = self.search_zus_website("ZUS Coffee test", num_results=1)
            
            return {
                "success": result["success"],
                "configured": bool(self.api_key and self.search_engine_id),
                "error": result.get("error"),
                "message": "Connection test completed"
            }
        except Exception as e:
            return {
                "success": False,
                "configured": bool(self.api_key and self.search_engine_id),
                "error": str(e),
                "message": "Connection test failed"
            }