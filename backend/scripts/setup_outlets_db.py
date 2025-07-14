"""
Script to setup ZUS Coffee outlets database
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from services.outlets_service import ZUSOutletsService
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_outlets_database():
    """Setup ZUS Coffee outlets database"""
    try:
        logger.info("Setting up ZUS Coffee outlets database...")
        
        # Initialize outlets service (this will create the database)
        outlets_service = ZUSOutletsService()
        
        logger.info("✅ Outlets database setup completed!")
        logger.info(f"Database location: {settings.OUTLETS_DB_PATH}")
        
        # Test the database
        logger.info("Testing database connection...")
        import sqlite3
        conn = sqlite3.connect(settings.OUTLETS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM outlets")
        count = cursor.fetchone()[0]
        conn.close()
        
        logger.info(f"✅ Database contains {count} outlets")
        
        return True
        
    except Exception as e:
        logger.error(f"Error setting up outlets database: {e}")
        return False

if __name__ == "__main__":
    success = setup_outlets_database()
    if success:
        print("✅ Outlets database setup completed successfully!")
    else:
        print("❌ Outlets database setup failed!")
        sys.exit(1)