from fastapi import APIRouter, HTTPException
from models.schemas import QueryRequest, QueryResponse
from tools.planner import AgenticPlanner
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["Agent"])

# Initialize planner ONCE (this maintains memory across requests)
planner = AgenticPlanner()

@router.post("/chat", response_model=QueryResponse)
async def chat_with_agent(request: QueryRequest):
    """
    Chat with the ZUS Coffee AI Agent
    
    This endpoint demonstrates agentic planning and tool calling.
    The agent can:
    - Answer general questions
    - Perform calculations  
    - Search ZUS Coffee products
    - Find ZUS Coffee outlets
    - Search the knowledge base
    
    Memory: YES - The agent remembers previous conversations
    """
    try:
        # Input validation
        if not request.message.strip():
            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty"
            )
        
        logger.info(f"Agent chat request: {request.message}")
        
        # Process with agent (memory is handled automatically by the planner)
        result = planner.plan_and_execute(request.message, request.chat_history)
        
        return QueryResponse(
            response=result["final_answer"],
            tools_used=result["tools_used"],
            decision_points=result["decision_points"],
            success=result["success"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in agent endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred while processing your request"
        )

@router.get("/memory")
async def get_agent_memory():
    """Get current agent memory for debugging"""
    try:
        messages = planner.memory.chat_memory.messages
        return {
            "memory_count": len(messages),
            "messages": [{"type": type(msg).__name__, "content": str(msg)[:100]} for msg in messages[-5:]]
        }
    except Exception as e:
        return {"error": str(e)}

@router.delete("/memory")
async def clear_agent_memory():
    """Clear agent memory"""
    try:
        planner.memory.clear()
        return {"message": "Agent memory cleared successfully"}
    except Exception as e:
        return {"error": str(e)}