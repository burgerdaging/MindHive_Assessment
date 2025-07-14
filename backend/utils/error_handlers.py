from fastapi import HTTPException
from models.schemas import ErrorResponse
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Centralized error handling for the application"""
    
    @staticmethod
    def handle_service_error(error: Exception, service_name: str) -> HTTPException:
        """Handle service-level errors and convert to HTTP exceptions"""
        error_msg = str(error)
        logger.error(f"Service error in {service_name}: {error_msg}")
        
        # Map specific errors to appropriate HTTP status codes
        if "not found" in error_msg.lower():
            return HTTPException(
                status_code=404,
                detail=f"{service_name} not found: {error_msg}"
            )
        elif "timeout" in error_msg.lower():
            return HTTPException(
                status_code=504,
                detail=f"{service_name} timeout: {error_msg}"
            )
        elif "unauthorized" in error_msg.lower() or "api key" in error_msg.lower():
            return HTTPException(
                status_code=401,
                detail=f"{service_name} authentication error: {error_msg}"
            )
        else:
            return HTTPException(
                status_code=500,
                detail=f"{service_name} internal error: {error_msg}"
            )
    
    @staticmethod
    def create_error_response(error_code: str, message: str, recovery_suggestions: list = None) -> ErrorResponse:
        """Create standardized error response"""
        return ErrorResponse(
            error=error_code,
            error_code=error_code,
            message=message,
            recovery_suggestions=recovery_suggestions or []
        )
    
    @staticmethod
    def handle_validation_error(error: Exception) -> HTTPException:
        """Handle validation errors"""
        logger.warning(f"Validation error: {error}")
        return HTTPException(
            status_code=400,
            detail=f"Validation error: {str(error)}"
        )
    
    @staticmethod
    def handle_security_error(error: Exception, attack_type: str = "unknown") -> HTTPException:
        """Handle security-related errors"""
        logger.warning(f"Security error ({attack_type}): {error}")
        return HTTPException(
            status_code=400,
            detail=f"Security violation detected: {attack_type}"
        )

def handle_service_error(error: Exception, service_name: str) -> HTTPException:
    """Convenience function for handling service errors"""
    return ErrorHandler.handle_service_error(error, service_name)