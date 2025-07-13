from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import sqlite3
from services.search_service import GoogleCustomSearchService
from config import settings
import logging
import os

logger = logging.getLogger(__name__)

class ZUSOutletsService:
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
        
        # MongoDB setup (for vector search)
        self.client = MongoClient(settings.MONGODB_ATLAS_CLUSTER_URI)
        self.collection = self.client[settings.DB_NAME][settings.ZUS_COLLECTION_NAME]
        self.vector_index_name = settings.ATLAS_VECTOR_SEARCH_INDEX_NAME
        
        # SQL Database setup
        self.db_path = settings.OUTLETS_DB_PATH
        self._ensure_db_directory()
        self._setup_outlets_database()
        
        # Search service
        self.search_service = GoogleCustomSearchService()
    
    def _ensure_db_directory(self):
        """Ensure database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def _setup_outlets_database(self):
        """Setup SQLite database with ZUS outlets data"""
        try:
            logger.info("Setting up ZUS outlets database...")
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS outlets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    address TEXT NOT NULL,
                    area TEXT NOT NULL,
                    phone TEXT,
                    hours TEXT,
                    services TEXT,
                    latitude REAL,
                    longitude REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Sample ZUS outlets data
            sample_outlets = [
                ("ZUS Coffee KLCC", "Lot G-23A, Ground Floor, Suria KLCC, Kuala Lumpur City Centre", "Kuala Lumpur", "+603-2382-2828", "8:00 AM - 10:00 PM", "Dine-in,Takeaway,Delivery", 3.1570, 101.7123),
                ("ZUS Coffee Pavilion KL", "Level 6, Pavilion Kuala Lumpur, Bukit Bintang", "Kuala Lumpur", "+603-2148-8833", "10:00 AM - 10:00 PM", "Dine-in,Takeaway", 3.1478, 101.7123),
                ("ZUS Coffee Mid Valley", "LG-074, Lower Ground, Mid Valley Megamall", "Kuala Lumpur", "+603-2287-3344", "10:00 AM - 10:00 PM", "Dine-in,Takeaway,Drive-thru", 3.1176, 101.6769),
                ("ZUS Coffee Sunway Pyramid", "LG2.72A, Lower Ground 2, Sunway Pyramid", "Selangor", "+603-7492-1122", "10:00 AM - 10:00 PM", "Dine-in,Takeaway", 3.0733, 101.6067),
                ("ZUS Coffee 1 Utama", "S330, 3rd Floor, 1 Utama Shopping Centre", "Selangor", "+603-7726-5566", "10:00 AM - 10:00 PM", "Dine-in,Takeaway,Delivery", 3.1502, 101.6154),
                ("ZUS Coffee IOI City Mall", "L1-35, Level 1, IOI City Mall, Putrajaya", "Selangor", "+603-8328-7788", "10:00 AM - 10:00 PM", "Dine-in,Takeaway", 2.9969, 101.7297),
                ("ZUS Coffee The Gardens Mall", "S-240, 2nd Floor, The Gardens Mall", "Kuala Lumpur", "+603-2282-9900", "10:00 AM - 10:00 PM", "Dine-in,Takeaway", 3.1176, 101.6769),
                ("ZUS Coffee Bangsar Village", "Ground Floor, Bangsar Village II", "Kuala Lumpur", "+603-2287-1234", "7:00 AM - 11:00 PM", "Dine-in,Takeaway,Drive-thru", 3.1205, 101.6711)
            ]
            
            # Check if data already exists
            cursor.execute("SELECT COUNT(*) FROM outlets")
            count = cursor.fetchone()[0]
            
            if count == 0:
                cursor.executemany("""
                    INSERT INTO outlets (name, address, area, phone, hours, services, latitude, longitude)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, sample_outlets)
                logger.info(f"Inserted {len(sample_outlets)} outlets into database")
            
            conn.commit()
            conn.close()
            logger.info("Outlets database setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up outlets database: {e}")
    
    async def search_outlets(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search ZUS outlets using vector store, SQL database, and live search"""
        try:
            logger.info(f"Searching ZUS outlets for: {query}")
            
            results = {
                "query": query,
                "vector_results": [],
                "sql_results": [],
                "live_search_results": [],
                "combined_summary": "",
                "success": True,
                "sources": []
            }
            
            # 1. Search vector store for outlet information
            try:
                vector_results = await self._search_vector_store(query, top_k)
                results["vector_results"] = vector_results
                if vector_results:
                    results["sources"].append("ZUS Document Store")
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                results["vector_results"] = []
            
            # 2. Try Text2SQL on local database
            try:
                sql_results = await self._query_sql_database(query)
                results["sql_results"] = sql_results
                if sql_results.get("results"):
                    results["sources"].append("SQL Database")
            except Exception as e:
                logger.warning(f"SQL search failed: {e}")
                results["sql_results"] = {"results": [], "sql_query": "", "success": False}
            
            # 3. Search live website for outlet information
            try:
                live_results = self._search_live_website(query)
                results["live_search_results"] = live_results
                if live_results:
                    results["sources"].append("Live Website")
            except Exception as e:
                logger.warning(f"Live search failed: {e}")
                results["live_search_results"] = []
            
            # 4. Generate combined summary
            results["combined_summary"] = await self._generate_combined_summary(query, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in outlet search: {e}")
            return {
                "query": query,
                "vector_results": [],
                "sql_results": {"results": [], "sql_query": "", "success": False},
                "live_search_results": [],
                "combined_summary": f"Search error: {str(e)}",
                "success": False,
                "sources": []
            }
    
    async def _search_vector_store(self, query: str, top_k: int) -> List[Dict]:
        """Search ZUS outlets in vector store"""
        try:
            enhanced_query = f"{query} ZUS Coffee outlet store location address branch"
            
            vector_search = MongoDBAtlasVectorSearch(
                embedding=self.embeddings,
                collection=self.collection,
                index_name=self.vector_index_name,
            )
            
            retriever = vector_search.as_retriever(search_kwargs={"k": top_k})
            relevant_docs = retriever.get_relevant_documents(enhanced_query)
            
            # Filter for outlet-related content
            outlet_results = []
            for doc in relevant_docs:
                content_lower = doc.page_content.lower()
                if any(word in content_lower for word in ["outlet", "store", "location", "address", "branch", "mall", "shopping"]):
                    outlet_results.append({
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "ZUS Document"),
                        "type": "stored_document"
                    })
            
            return outlet_results
            
        except Exception as e:
            logger.error(f"Vector store search error: {e}")
            return []
    
    async def _query_sql_database(self, nl_query: str) -> Dict[str, Any]:
        """Convert natural language to SQL and execute"""
        try:
            # Generate SQL query using LLM
            sql_prompt = f"""
            Convert this natural language query to SQL for the outlets database:
            
            Database Schema:
            Table: outlets
            - id (INTEGER): Unique identifier
            - name (TEXT): Outlet name (e.g., "ZUS Coffee KLCC")
            - address (TEXT): Full address
            - area (TEXT): Area/City (Kuala Lumpur, Selangor, etc.)
            - phone (TEXT): Phone number
            - hours (TEXT): Operating hours (e.g., "8:00 AM - 10:00 PM")
            - services (TEXT): Services (comma-separated: Dine-in,Takeaway,Delivery,Drive-thru)
            - latitude, longitude (REAL): Coordinates
            
            Query: "{nl_query}"
            
            Rules:
            1. Only use SELECT statements
            2. Use proper SQL syntax for SQLite
            3. For location queries, search in both 'area' and 'address' fields using LIKE
            4. For service queries, use LIKE operator since services are comma-separated
            5. Return only the SQL query, no explanations
            
            SQL Query:
            """
            
            response = self.llm.invoke(sql_prompt)
            sql_query = response.content.strip().replace("```sql", "").replace("```", "").strip()
            
            # Execute SQL query
            results = self._execute_sql_query(sql_query)
            
            return {
                "sql_query": sql_query,
                "results": results,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Text2SQL error: {e}")
            return {
                "sql_query": "",
                "results": [],
                "success": False,
                "error": str(e)
            }
    
    def _execute_sql_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query safely"""
        try:
            # Security: Basic SQL injection protection
            dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter', 'create', '--', ';']
            query_lower = sql_query.lower()
            
            for keyword in dangerous_keywords:
                if keyword in query_lower:
                    raise ValueError(f"Potentially dangerous SQL keyword detected: {keyword}")
            
            # Ensure it's a SELECT query
            if not query_lower.strip().startswith('select'):
                raise ValueError("Only SELECT queries are allowed")
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            
            results = [dict(row) for row in rows]
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return []
    
    def _search_live_website(self, query: str) -> List[Dict]:
        """Search live ZUS website for outlets"""
        try:
            enhanced_query = f"{query} ZUS Coffee store outlet location branch"
            search_results = self.search_service.search_zus_website(enhanced_query, num_results=5)
            
            if not search_results["success"]:
                return []
            
            outlet_results = []
            for result in search_results["results"]:
                if result["category"] in ["outlet", "general"]:
                    outlet_results.append({
                        "title": result["title"],
                        "url": result["link"],
                        "description": result["snippet"],
                        "category": result["category"],
                        "type": "live_search"
                    })
            
            return outlet_results
            
        except Exception as e:
            logger.error(f"Live search error: {e}")
            return []
    
    async def _generate_combined_summary(self, query: str, results: Dict) -> str:
        """Generate AI summary combining all sources"""
        try:
            context_parts = []
            
            if results["vector_results"]:
                context_parts.append("=== ZUS Outlet Documentation ===")
                for result in results["vector_results"]:
                    context_parts.append(result["content"])
            
            if results["sql_results"].get("results"):
                context_parts.append("\n=== Database Results ===")
                for result in results["sql_results"]["results"]:
                    outlet_info = f"Name: {result.get('name', 'N/A')}\nAddress: {result.get('address', 'N/A')}\nArea: {result.get('area', 'N/A')}\nHours: {result.get('hours', 'N/A')}\nServices: {result.get('services', 'N/A')}\nPhone: {result.get('phone', 'N/A')}"
                    context_parts.append(outlet_info)
            
            if results["live_search_results"]:
                context_parts.append("\n=== Live Website Results ===")
                for result in results["live_search_results"]:
                    context_parts.append(f"Title: {result['title']}\nDescription: {result['description']}\nURL: {result['url']}")
            
            if not context_parts:
                return "No relevant ZUS Coffee outlet information found for your query."
            
            combined_context = "\n\n".join(context_parts)
            
            summary_prompt = f"""
            Based on the following information about ZUS Coffee outlets, provide a comprehensive and helpful summary for the user's query: "{query}"

            Available Information:
            {combined_context}

            Instructions:
            1. Provide a clear, helpful summary that directly addresses the user's query about ZUS outlets
            2. Include specific details like addresses, operating hours, and services when available
            3. If multiple outlets are relevant, organize them clearly
            4. Include contact information and directions when available
            5. Keep the response customer-focused and actionable
            6. If live website links are available, mention them

            Summary:
            """
            
            response = self.llm.invoke(summary_prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Unable to generate summary: {str(e)}"