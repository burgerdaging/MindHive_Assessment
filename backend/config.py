import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
    SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
    SEARCH_PUBLIC_URL = os.getenv("SEARCH_PUBLIC_URL")
    
    # MongoDB Configuration
    MONGODB_ATLAS_CLUSTER_URI = os.getenv("MONGODB_ATLAS_CLUSTER_URI")
    DB_NAME = "MongoDB"
    COLLECTION_NAME = "Facts-txt"
    ZUS_COLLECTION_NAME = "Zus-Coffee-Document.pdf"  # Make sure this collection exists
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "MindHive-Assessment"  # Make sure this index exists
    
    # Application Settings
    APP_NAME = "ZUS Coffee AI Assistant"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Security Settings
    MAX_QUERY_LENGTH = 500
    RATE_LIMIT_PER_MINUTE = 60
    
    # Database Settings
    OUTLETS_DB_PATH = "data/zus_outlets.db"

settings = Settings()