from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime

class QueryRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, description="User query message")
    chat_history: List[Dict] = Field(default=[], description="Previous conversation history")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty or whitespace only')
        return v.strip()

class QueryResponse(BaseModel):
    response: str
    tools_used: List[str] = []
    decision_points: List[str] = []
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)
    chat_history: List[Dict] = Field(default=[], description="Updated conversation history")

class ProductQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Product search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class ProductQueryResponse(BaseModel):
    summary: str
    products: List[Dict[str, Any]]
    query: str
    total_found: int
    success: bool = True
    sources: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.now)

class OutletQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Outlet search query")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        # Basic SQL injection prevention
        dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter', 'create', '--', ';']
        query_lower = v.lower()
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                raise ValueError(f'Query contains potentially dangerous keyword: {keyword}')
        return v.strip()

class OutletQueryResponse(BaseModel):
    summary: str
    outlets: List[Dict[str, Any]]
    query: str
    sql_query: Optional[str] = None
    total_found: int
    success: bool = True
    sources: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    error: str
    error_code: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    recovery_suggestions: List[str] = []