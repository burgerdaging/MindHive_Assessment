# ZUS Coffee AI Assistant

An intelligent AI assistant with agentic planning capabilities, specialized for ZUS Coffee product and outlet information.

## Features

- **Agentic Planning**: Intelligent decision-making and tool selection
- **Product Search**: Vector-based search through ZUS Coffee drinkware products
- **Outlet Search**: Text2SQL conversion for finding ZUS Coffee outlets
- **Live Website Search**: Real-time search through ZUS Coffee website
- **Error Handling**: Robust error handling and security measures
- **API Documentation**: Complete OpenAPI specification

## Project Structure
zus_coffee_ai/
├── main.py                     # Main FastAPI application
├── config.py                   # Configuration settings
├── models/schemas.py           # Pydantic models
├── services/                   # Business logic services
├── api/                        # FastAPI route handlers
├── tools/                      # LangChain tools and planner
├── utils/                      # Utility functions
├── tests/                      # Test suite
├── scripts/                    # Setup scripts
└── requirements.txt

## SETUP INSTRUCTIONS

## 1. Install Dependencies
   pip install -r requirements.txt

## 2. Environment Configuration
    Add/Edit .env file according to your API Keys

## 3. Setup Databases
    python scripts/setup_outlets_db.py
    python scripts/ingest_products.py
    
## 4. Run Applications
    python main.py



API Endpoints

Products Endpoint

GET /products?query=<user_question>
POST /products/search

Outlets Endpoint

GET /outlets?query=<nl_query>
POST /outlets/search

Agent Endpoint

POST /agent/chat

Health Checks

GET /health
GET /products/health
GET /outlets/health
GET /agent/health