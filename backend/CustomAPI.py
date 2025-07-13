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
import requests
import json
from urllib.parse import quote_plus
import re
from typing import List, Dict, Optional

#Part 4 & 5 endpoint is here 

# Add these imports to your existing code
import requests
import json
from urllib.parse import quote_plus
import re
from typing import List, Dict, Optional

# ============= GOOGLE CUSTOM SEARCH ENGINE INTEGRATION =============

class GoogleCustomSearchTool:
    def __init__(self):
        # You'll need to get these from Google Cloud Console
        self.api_key = os.getenv("GOOGLE_SEARCH_API_KEY")  # Add to your .env
        self.search_engine_id = "b6fecb64edd314067"  # Your CSE ID
        self.base_url = "https://cse.google.com/cse.js?cx=b6fecb64edd314067"
        
    def search_zus_website(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """Search ZUS Coffee website using Google Custom Search"""
        try:
            print(f"🔍 Searching ZUS website for: {query}")
            
            # Prepare search parameters
            params = {
                'key': self.api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10),  # Max 10 per request
                'safe': 'active'
            }
            
            # Make API request
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            search_data = response.json()
            
            # Process search results
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
                # Extract relevant information
                result = {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "display_link": item.get("displayLink", ""),
                }
                
                # Add category detection
                result["category"] = self._detect_category(item)
                
                # Clean and enhance snippet
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
        
        # Remove extra whitespace and clean up
        cleaned = re.sub(r'\s+', ' ', snippet.strip())
        
        # Remove common unwanted patterns
        cleaned = re.sub(r'^\d+\s*[.-]\s*', '', cleaned)  # Remove leading numbers
        
        return cleaned

# Initialize search tool
google_search_tool = GoogleCustomSearchTool()

