from fastapi import APIRouter, HTTPException, Query
from models.schemas import ProductQueryRequest, ProductQueryResponse, ErrorResponse
from services.products_service import ZUSProductsService
from utils.error_handlers import handle_service_error
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/products", tags=["Products"])

# Initialize service
products_service = ZUSProductsService()

@router.get("/", response_model=ProductQueryResponse)
async def search_products(
    query: str = Query(..., min_length=1, max_length=500, description="Product search query"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results to return")
):
    """
    Search ZUS Coffee drinkware products
    
    This endpoint searches through ZUS Coffee's drinkware products using:
    - Vector search on ingested product documents
    - Live website search for current information
    
    Returns an AI-generated summary with relevant product information.
    """
    try:
        # Input validation
        if not query.strip():
            raise HTTPException(
                status_code=400, 
                detail="Query cannot be empty"
            )
        
        # Search products
        result = await products_service.search_products(query.strip(), top_k)
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Product search failed: {result.get('combined_summary', 'Unknown error')}"
            )
        
        # Combine results for response
        all_products = []
        all_products.extend(result["vector_results"])
        all_products.extend(result["live_search_results"])
        
        return ProductQueryResponse(
            summary=result["combined_summary"],
            products=all_products,
            query=result["query"],
            total_found=len(all_products),
            success=result["success"],
            sources=result["sources"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in products endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while searching products"
        )

@router.post("/search", response_model=ProductQueryResponse)
async def search_products_post(request: ProductQueryRequest):
    """
    Search ZUS Coffee drinkware products (POST method)
    
    Alternative POST endpoint for product search with request body.
    """
    try:
        result = await products_service.search_products(request.query, request.top_k)
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Product search failed: {result.get('combined_summary', 'Unknown error')}"
            )
        
        all_products = []
        all_products.extend(result["vector_results"])
        all_products.extend(result["live_search_results"])
        
        return ProductQueryResponse(
            summary=result["combined_summary"],
            products=all_products,
            query=result["query"],
            total_found=len(all_products),
            success=result["success"],
            sources=result["sources"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in products POST endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while searching products"
        )

@router.get("/health")
async def products_health_check():
    """Health check endpoint for products service"""
    try:
        # Test basic functionality
        test_result = await products_service.search_products("test", 1)
        return {
            "status": "healthy",
            "service": "products",
            "vector_store_available": len(test_result.get("vector_results", [])) >= 0,
            "live_search_available": test_result.get("sources", []) != []
        }
    except Exception as e:
        logger.error(f"Products health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "products",
            "error": str(e)
        }