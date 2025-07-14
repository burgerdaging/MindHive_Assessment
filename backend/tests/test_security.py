import pytest
from utils.security import SecurityValidator

class TestSecurity:
    """Security validation tests"""
    
    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection"""
        # Safe queries
        safe_queries = [
            "SELECT * FROM outlets WHERE area = 'Kuala Lumpur'",
            "SELECT name, address FROM outlets WHERE id = 1",
            "SELECT * FROM outlets WHERE services LIKE '%Delivery%'"
        ]
        
        for query in safe_queries:
            result = SecurityValidator.validate_sql_query(query)
            assert result["is_safe"] == True
        
        # Dangerous queries
        dangerous_queries = [
            "DROP TABLE outlets",
            "DELETE FROM outlets WHERE 1=1",
            "SELECT * FROM outlets; DROP TABLE users; --",
            "INSERT INTO outlets VALUES ('hack', 'hack')",
            "UPDATE outlets SET name = 'hacked'",
            "SELECT * FROM outlets WHERE 1=1 OR 1=1"
        ]
        
        for query in dangerous_queries:
            result = SecurityValidator.validate_sql_query(query)
            assert result["is_safe"] == False
            assert result["threat_type"] in ["sql_injection", "unauthorized_operation"]
    
    def test_xss_detection(self):
        """Test XSS pattern detection"""
        # Safe inputs
        safe_inputs = [
            "What are ZUS Coffee products?",
            "Find outlets in Kuala Lumpur",
            "Calculate 2 + 2",
            "Tell me about coffee"
        ]
        
        for input_text in safe_inputs:
            result = SecurityValidator.validate_user_input(input_text)
            assert result["is_safe"] == True
        
        # XSS attempts
        xss_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('hack')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert(1)'></iframe>",
            "<div onload=alert('xss')>content</div>"
        ]
        
        for input_text in xss_inputs:
            result = SecurityValidator.validate_user_input(input_text)
            assert result["is_safe"] == False
            assert any(threat["type"] == "xss" for threat in result["threats"])
    
    def test_command_injection_detection(self):
        """Test command injection pattern detection"""
        # Command injection attempts
        command_injections = [
            "test; rm -rf /",
            "query | cat /etc/passwd",
            "input && shutdown -h now",
            "search `whoami`",
            "find $(id)",
            "query; del *.*"
        ]
        
        for input_text in command_injections:
            result = SecurityValidator.validate_user_input(input_text)
            assert result["is_safe"] == False
            assert any(threat["type"] == "command_injection" for threat in result["threats"])
    
    def test_input_sanitization(self):
        """Test input sanitization"""
        test_cases = [
            {
                "input": "<script>alert('xss')</script>Hello",
                "expected_contains": "Hello",
                "expected_not_contains": "<script>"
            },
            {
                "input": "javascript:alert('hack')Normal text",
                "expected_contains": "Normal text",
                "expected_not_contains": "javascript:"
            },
            {
                "input": "<div>Content</div>",
                "expected_contains": "Content",
                "expected_not_contains": "<div>"
            }
        ]
        
        for case in test_cases:
            sanitized = SecurityValidator.sanitize_input(case["input"])
            assert case["expected_contains"] in sanitized
            assert case["expected_not_contains"] not in sanitized
    
    def test_excessive_length_detection(self):
        """Test detection of excessively long inputs"""
        # Normal length input
        normal_input = "A" * 100
        result = SecurityValidator.validate_user_input(normal_input)
        assert result["is_safe"] == True
        
        # Excessively long input
        long_input = "A" * 15000
        result = SecurityValidator.validate_user_input(long_input)
        assert result["is_safe"] == False
        assert any(threat["type"] == "excessive_length" for threat in result["threats"])