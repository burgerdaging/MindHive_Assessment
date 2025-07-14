# from typing import Dict, List, Any
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
# from langchain_mongodb import MongoDBAtlasVectorSearch
# from pymongo import MongoClient
# from services.search_service import GoogleCustomSearchService
# from config import settings
# import logging

# async def check_vector_store(self):
#     """Diagnostic method to check vector store contents"""
#     try:
#         # Check collection count
#         count = self.collection.count_documents({})
#         logger.info(f"Vector store contains {count} documents")
        
#         # Check a sample document
#         sample = self.collection.find_one({})
#         if sample:
#             logger.info(f"Sample document content: {sample.get('page_content','')[:200]}...")
#             logger.info(f"Sample metadata: {sample.get('metadata', {})}")
#         else:
#             logger.warning("No documents found in vector store")
            
#         return {
#             "document_count": count,
#             "sample_document": sample
#         }
#     except Exception as e:
#         logger.error(f"Vector store check failed: {e}")
#         raise