import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.products import router as products_router
from api.outlets import router as outlets_router
from api.agent import router as agent_router
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI Assistant with Agentic Planning and ZUS Coffee Integration",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "recovery_suggestions": [
                "Check your input and try again",
                "Contact support if the issue persists",
                "Try a different query or endpoint"
            ]
        }
    )

# Include routers
app.include_router(products_router)
app.include_router(outlets_router)
app.include_router(agent_router)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "ZUS Coffee AI Assistant API",
        "version": settings.APP_VERSION,
        "endpoints": {
            "products": "/products?query=<your_question>",
            "outlets": "/outlets?query=<your_question>",
            "agent": "/agent/chat (POST)",
            "docs": "/docs",
            "health": "/health"
        },
        "features": [
            "Agentic Planning and Tool Calling",
            "ZUS Coffee Product Search",
            "ZUS Coffee Outlet Search with Text2SQL",
            "Vector Store Integration",
            "Live Website Search",
            "Error Handling and Security"
        ]
    }

@app.get("/health")
async def health_check():
    """Global health check endpoint"""
    try:
        return {
            "status": "healthy",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "services": {
                "products": "/products/health",
                "outlets": "/outlets/health", 
                "agent": "/agent/health"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )