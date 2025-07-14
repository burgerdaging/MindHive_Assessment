import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SecurityValidator:
    """Security validation utilities"""
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(or|and)\s+\d+\s*=\s*\d+)",
        r"(\b(or|and)\s+['\"].*['\"])",
        r"(;|\|\||&&)",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
    ]
    
    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        r"(\||&|;|`|\$\(|\${)",
        r"(rm|del|format|shutdown|reboot)",
        r"(cat|type|more|less|head|tail)",
    ]
    
    @classmethod
    def validate_sql_query(cls, query: str) -> Dict[str, Any]:
        """Validate SQL query for potential injection attacks"""
        query_lower = query.lower()
        
        # Check for dangerous keywords
        dangerous_keywords = ['drop', 'delete', 'update', 'insert', 'alter', 'create', '--', ';']
        found_keywords = [keyword for keyword in dangerous_keywords if keyword in query_lower]
        
        if found_keywords:
            return {
                "is_safe": False,
                "threat_type": "sql_injection",
                "details": f"Dangerous keywords found: {found_keywords}",
                "severity": "high"
            }
        
        # Check for SQL injection patterns
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return {
                    "is_safe": False,
                    "threat_type": "sql_injection",
                    "details": f"SQL injection pattern detected: {pattern}",
                    "severity": "high"
                }
        
        # Ensure it's a SELECT query
        if not query_lower.strip().startswith('select'):
            return {
                "is_safe": False,
                "threat_type": "unauthorized_operation",
                "details": "Only SELECT queries are allowed",
                "severity": "medium"
            }
        
        return {
            "is_safe": True,
            "threat_type": None,
            "details": "Query appears safe",
            "severity": "none"
        }
    
    @classmethod
    def validate_user_input(cls, user_input: str) -> Dict[str, Any]:
        """Validate user input for various security threats"""
        threats_found = []
        
        # Check for XSS
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                threats_found.append({
                    "type": "xss",
                    "pattern": pattern,
                    "severity": "high"
                })
        
        # Check for command injection
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                threats_found.append({
                    "type": "command_injection",
                    "pattern": pattern,
                    "severity": "high"
                })
        
        # Check input length
        if len(user_input) > 10000:
            threats_found.append({
                "type": "excessive_length",
                "pattern": "input_too_long",
                "severity": "medium"
            })
        
        return {
            "is_safe": len(threats_found) == 0,
            "threats": threats_found,
            "threat_count": len(threats_found)
        }
    
    @classmethod
    def sanitize_input(cls, user_input: str) -> str:
        """Sanitize user input by removing potentially dangerous content"""
        # Remove HTML tags
        sanitized = re.sub(r'<[^>]+>', '', user_input)
        
        # Remove script tags
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove javascript: protocols
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        
        # Limit length
        if len(sanitized) > 1000:
            sanitized = sanitized[:1000] + "..."
        
        return sanitized.strip()