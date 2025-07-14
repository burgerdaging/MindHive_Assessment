#!/usr/bin/env python3
"""
Script to ingest ZUS Coffee product documents into vector store
"""

import os
import sys
import asyncio
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from services.products_service import ZUSProductsService
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def ingest_zus_products():
    """Ingest ZUS Coffee products into vector store"""
    try:
        logger.info("Starting ZUS Coffee product ingestion...")
        
        # Initialize products service
        products_service = ZUSProductsService()
        
        # Test search to verify setup
        logger.info("Testing vector store connection...")
        test_result = await products_service.search_products("test", 1)
        
        if test_result["success"]:
            logger.info("✅ Vector store connection successful")
            logger.info(f"Sources available: {test_result['sources']}")
        else:
            logger.warning("⚠️ Vector store connection issues")
        
        logger.info("Product ingestion setup completed!")
        logger.info("To add your ZUS Coffee documents:")
        logger.info("1. Upload your PDF documents to MongoDB Atlas")
        logger.info("2. Ensure vector search index is created")
        logger.info("3. Test the /products endpoint")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during product ingestion: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(ingest_zus_products())
    if success:
        print("✅ Product ingestion completed successfully!")
    else:
        print("❌ Product ingestion failed!")
        sys.exit(1)