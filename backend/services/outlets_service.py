from typing import Dict, List, Any
from langchain_google_genai import ChatGoogleGenerativeAI
import sqlite3
from services.search_service import GoogleCustomSearchService
from config import settings
import logging
import os

logger = logging.getLogger(__name__)

class ZUSOutletsService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            api_key=settings.GEMINI_API_KEY, 
            temperature=0.0
        )
        
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
            
            # Check if data already exists
            cursor.execute("SELECT COUNT(*) FROM outlets")
            count = cursor.fetchone()[0]
            
            if count == 0:
                # Sample ZUS outlets data
                sample_outlets = [
                    ("ZUS Coffee KLCC", "Lot G-23A, Ground Floor, Suria KLCC, Kuala Lumpur City Centre", "Kuala Lumpur", "+603-2382-2828", "8:00 AM - 10:00 PM", "Dine-in,Takeaway,Delivery", 3.1570, 101.7123),
                    ("ZUS Coffee Pavilion KL", "Level 6, Pavilion Kuala Lumpur, Bukit Bintang", "Kuala Lumpur", "+603-2148-8833", "10:00 AM - 10:00 PM", "Dine-in,Takeaway", 3.1478, 101.7123),
                    ("ZUS Coffee Mid Valley", "LG-074, Lower Ground, Mid Valley Megamall", "Kuala Lumpur", "+603-2287-3344", "10:00 AM - 10:00 PM", "Dine-in,Takeaway,Drive-thru", 3.1176, 101.6769),
                    ("ZUS Coffee Sunway Pyramid", "LG2.72A, Lower Ground 2, Sunway Pyramid", "Selangor", "+603-7492-1122", "10:00 AM - 10:00 PM", "Dine-in,Takeaway", 3.0733, 101.6067),
                    ("ZUS Coffee 1 Utama", "S330, 3rd Floor, 1 Utama Shopping Centre", "Selangor", "+603-7726-5566", "10:00 AM - 10:00 PM", "Dine-in,Takeaway,Delivery", 3.1502, 101.6154),
                    ("ZUS Coffee IOI City Mall", "L1-35, Level 1, IOI City Mall, Putrajaya", "Selangor", "+603-8328-7788", "10:00 AM - 10:00 PM", "Dine-in,Takeaway", 2.9969, 101.7297),
                    ("ZUS Coffee The Gardens Mall", "S-240, 2nd Floor, The Gardens Mall", "Kuala Lumpur", "+603-2282-9900", "10:00 AM - 10:00 PM", "Dine-in,Takeaway", 3.1176, 101.6769),
                    ("ZUS Coffee Bangsar Village", "Ground Floor, Bangsar Village II", "Kuala Lumpur", "+603-2287-1234", "7:00 AM - 11:00 PM", "Dine-in,Takeaway,Drive-thru", 3.1205, 101.6711),
                    ("ZUS Coffee Ampang Point", "Level 2, Ampang Point Shopping Centre", "Selangor", "+603-4270-5566", "10:00 AM - 10:00 PM", "Dine-in,Takeaway", 3.1502, 101.7654),
                    ("ZUS Coffee Subang Jaya", "Ground Floor, SS15 Subang Jaya", "Selangor", "+603-5633-7788", "7:00 AM - 11:00 PM", "Dine-in,Takeaway,Drive-thru", 3.0738, 101.5183)
                ]
                
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
    
    async def _generate_combined_summary(self, query: str, results: Dict[str, Any]) -> str:
        """Generate AI summary combining SQL and live search results"""
        try:
            context_parts = []

            if results["sql_results"].get("results"):
                context_parts.append("===Database Results===")
                for outlet in results["sql_results"]["results"]:
                    context_parts.append(
                        f"{outlet['name']}\n"
                        f"Address: {outlet['address']}\n"
                        f"Hours: {outlet['hours']}\n"
                        f"Services: {outlet['services']}"
                    )
            if not context_parts:
                return f"I couldn't find any outlets matching your query: {query}"
            
            summary_prompt = f"""Based on the information provided about Zus Coffee outlets,
            provide a helpful response to the user's query: "{query}"
            
            Avaialable Information:
            {context_parts}

            Instructions:
            1. Provide a clear, helpful summary that directly addresses the user's query
            2. Include specific details like location, hours, and services when available
            3. If you have both database and live information, combine them intelligently
            4. Keep the response customer-focused and actionable
            5. If live website links are available, mention them for current information
            6. If information is limited, acknowledge this and suggest where to find more details

            Summary Response:
            """
            response = self.llm.invoke(summary_prompt)
            return response.content.strip()
        
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"I found some outlet information but couldn't generate a proper summary. Error: {str(e)}"

    async def search_outlets(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search ZUS outlets using SQL database and live search"""
        try:
            logger.info(f"Searching ZUS outlets for: {query}")
        
            results = {
                "query": query,
                "sql_results": [],
                "live_search_results": [],
                "combined_summary": "",
                "success": True,
                "sources": []
            }
            
            # 1. Try SQL database first
            try:
                sql_results = await self._query_sql_database(query)
                results["sql_results"] = sql_results
                if sql_results.get("results"):
                    results["sources"].append("SQL Database")
                    logger.info(f"Found {len(sql_results['results'])} results in SQL database")
            except Exception as e:
                logger.warning(f"SQL search failed: {e}")
                results["sql_results"] = {"results": [], "sql_query": "", "success": False}
              
            # 3. Generate combined summary - THIS IS THE FIXED LINE
            results["combined_summary"] = await self._generate_combined_summary(query, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in outlet search: {e}")
            return {
                "query": query,
                "sql_results": {"results": [], "sql_query": "", "success": False},
                "live_search_results": [],
                "combined_summary": f"Search error: {str(e)}",
                "success": False,
                "sources": []
            }
    
    async def _query_sql_database(self, nl_query: str) -> Dict[str, Any]:
        """
        ENHANCED TEXT2SQL IMPLEMENTATION
        Convert natural language to SQL and execute - THIS SOLVES YOUR TEXT2SQL OBJECTIVE
        """
        try:
            logger.info(f"Converting natural language to SQL: {nl_query}")
            
            # Try enhanced LLM-based SQL generation first
            sql_query = await self._generate_enhanced_sql(nl_query)
            
            # Fallback to simple keyword-based if LLM fails
            if not sql_query or "error" in sql_query.lower():
                logger.info("LLM SQL generation failed, using keyword-based fallback")
                sql_query = self._generate_simple_sql(nl_query)
            
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
    
    async def _generate_enhanced_sql(self, nl_query: str) -> str:
        """
        ENHANCED LLM-BASED TEXT2SQL GENERATION
        Uses Gemini to convert natural language to SQL with better accuracy
        """
        try:
            sql_prompt = f"""
            You are a SQL expert. Convert the following natural language query to a SQL query for the ZUS Coffee outlets database.
            
            Database Schema:
            Table: outlets
            - id (INTEGER): Unique identifier
            - name (TEXT): Outlet name (e.g., "ZUS Coffee KLCC", "ZUS Coffee Pavilion KL")
            - address (TEXT): Full address (e.g., "Lot G-23A, Ground Floor, Suria KLCC")
            - area (TEXT): Area/City ("Kuala Lumpur" or "Selangor")
            - phone (TEXT): Phone number (e.g., "+603-2382-2828")
            - hours (TEXT): Operating hours (e.g., "8:00 AM - 10:00 PM")
            - services (TEXT): Services comma-separated ("Dine-in,Takeaway,Delivery,Drive-thru")
            - latitude, longitude (REAL): GPS coordinates
            
            Natural Language Query: "{nl_query}"
            
            Rules:
            1. Only use SELECT statements
            2. Use proper SQL syntax for SQLite
            3. For location queries, search in both 'area' and 'address' fields using LIKE with wildcards
            4. For service queries, use LIKE operator since services are comma-separated
            5. Use LIMIT 10 to prevent too many results
            6. Use ILIKE or LIKE with % wildcards for partial matching
            7. Return only the SQL query, no explanations or markdown
            
            Examples:
            "outlets in Kuala Lumpur" → SELECT * FROM outlets WHERE area LIKE '%Kuala Lumpur%' LIMIT 10
            "stores with delivery" → SELECT * FROM outlets WHERE services LIKE '%Delivery%' LIMIT 10
            "ZUS Coffee KLCC" → SELECT * FROM outlets WHERE name LIKE '%KLCC%' LIMIT 10
            "outlets in KL" → SELECT * FROM outlets WHERE area LIKE '%Kuala Lumpur%' LIMIT 10
            "stores open early" → SELECT * FROM outlets WHERE hours LIKE '%7:00 AM%' LIMIT 10
            "outlets with drive thru" → SELECT * FROM outlets WHERE services LIKE '%Drive-thru%' LIMIT 10
            "all outlets" → SELECT * FROM outlets ORDER BY name LIMIT 10
            "outlets in Selangor" → SELECT * FROM outlets WHERE area LIKE '%Selangor%' LIMIT 10
            
            SQL Query:
            """
            
            response = self.llm.invoke(sql_prompt)
            sql_query = response.content.strip()
            
            # Clean up the response
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            # Remove any explanatory text that might be included
            lines = sql_query.split('\n')
            sql_lines = [line for line in lines if line.strip().upper().startswith('SELECT')]
            if sql_lines:
                sql_query = sql_lines[0].strip()
            
            logger.info(f"LLM Generated SQL: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Enhanced SQL generation error: {e}")
            return ""
    
    def _generate_simple_sql(self, query: str) -> str:
        """
        FALLBACK: Generate SQL query using simple keyword matching
        This is your existing implementation as a backup
        """
        query_lower = query.lower()
        
        # Base query
        sql = "SELECT * FROM outlets"
        conditions = []
        
        # Location-based conditions
        if "kuala lumpur" in query_lower or "kl" in query_lower:
            conditions.append("area LIKE '%Kuala Lumpur%'")
        elif "selangor" in query_lower:
            conditions.append("area LIKE '%Selangor%'")
        
        # Specific location searches
        locations = ["klcc", "pavilion", "mid valley", "sunway", "utama", "bangsar", "ampang", "subang"]
        for location in locations:
            if location in query_lower:
                conditions.append(f"(name LIKE '%{location}%' OR address LIKE '%{location}%')")
                break
        
        # Service-based conditions
        if "delivery" in query_lower:
            conditions.append("services LIKE '%Delivery%'")
        elif "drive" in query_lower or "thru" in query_lower:
            conditions.append("services LIKE '%Drive-thru%'")
        elif "takeaway" in query_lower:
            conditions.append("services LIKE '%Takeaway%'")
        
        # Hours-based conditions
        if "early" in query_lower or "7" in query_lower:
            conditions.append("hours LIKE '%7:00 AM%'")
        elif "late" in query_lower or "11" in query_lower:
            conditions.append("hours LIKE '%11:00 PM%'")
        
        # Add conditions to query
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += " ORDER BY name LIMIT 10"
        
        logger.info(f"Keyword-based Generated SQL: {sql}")
        return sql
    
    def _execute_sql_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute SQL query safely with enhanced security"""
        try:
            # Enhanced Security: SQL injection protection
            dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter', 'create', '--', ';', 'union', 'exec']
            query_lower = sql_query.lower()
            
            for keyword in dangerous_keywords:
                if keyword in query_lower and keyword not in ['select', 'from', 'where', 'like', 'and', 'or', 'order', 'by', 'limit']:
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
            
            logger.info(f"SQL query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"SQL execution error: {e}")