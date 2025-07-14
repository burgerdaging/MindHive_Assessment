from fastapi import APIRouter, HTTPException, Query
from models.schemas import OutletQueryRequest, OutletQueryResponse, ErrorResponse
from services.outlets_service import ZUSOutletsService
from utils.error_handlers import handle_service_error
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/outlets", tags=["Outlets"])

# Initialize service
outlets_service = ZUSOutletsService()

@router.get("/", response_model=OutletQueryResponse)
async def search_outlets(
    query: str = Query(..., min_length=1, max_length=500, description="Outlet search query")
):
    """
    Search ZUS Coffee outlet locations
    
    This endpoint searches through ZUS Coffee outlets using:
    - Text2SQL conversion for structured queries
    - Vector search on outlet documents
    - Live website search for current information
    
    Returns an AI-generated summary with relevant outlet information.
    """
    try:
        # Input validation and security check
        if not query.strip():
            raise HTTPException(
                status_code=400, 
                detail="Query cannot be empty"
            )
        
        # Basic SQL injection prevention
        dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter', 'create', '--', ';']
        query_lower = query.lower()
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                raise HTTPException(
                    status_code=400,
                    detail=f"Query contains potentially dangerous keyword: {keyword}"
                )
        
        # Search outlets
        result = await outlets_service.search_outlets(query.strip())
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Outlet search failed: {result.get('combined_summary', 'Unknown error')}"
            )
        
        # Combine results for response
        all_outlets = []
        all_outlets.extend(result["vector_results"])
        all_outlets.extend(result["sql_results"].get("results", []))
        all_outlets.extend(result["live_search_results"])
        
        return OutletQueryResponse(
            summary=result["combined_summary"],
            outlets=all_outlets,
            query=result["query"],
            sql_query=result["sql_results"].get("sql_query"),
            total_found=len(all_outlets),
            success=result["success"],
            sources=result["sources"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in outlets endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while searching outlets"
        )

@router.post("/search", response_model=OutletQueryResponse)
async def search_outlets_post(request: OutletQueryRequest):
    """
    Search ZUS Coffee outlet locations (POST method)
    
    Alternative POST endpoint for outlet search with request body.
    """
    try:
        result = await outlets_service.search_outlets(request.query)
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Outlet search failed: {result.get('combined_summary', 'Unknown error')}"
            )
        
        all_outlets = []
        all_outlets.extend(result["vector_results"])
        all_outlets.extend(result["sql_results"].get("results", []))
        all_outlets.extend(result["live_search_results"])
        
        return OutletQueryResponse(
            summary=result["combined_summary"],
            outlets=all_outlets,
            query=result["query"],
            sql_query=result["sql_results"].get("sql_query"),
            total_found=len(all_outlets),
            success=result["success"],
            sources=result["sources"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in outlets POST endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while searching outlets"
        )

@router.get("/health")
async def outlets_health_check():
    """Health check endpoint for outlets service"""
    try:
        # Test basic functionality
        test_result = await outlets_service.search_outlets("test")
        return {
            "status": "healthy",
            "service": "outlets",
            "sql_database_available": test_result.get("sql_results", {}).get("success", False),
            "vector_store_available": len(test_result.get("vector_results", [])) >= 0,
            "live_search_available": test_result.get("sources", []) != []
        }
    except Exception as e:
        logger.error(f"Outlets health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "outlets",
            "error": str(e)
        }