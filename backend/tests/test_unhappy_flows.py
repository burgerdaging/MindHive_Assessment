import pytest
import asyncio
from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch, MagicMock

client = TestClient(app)

class TestUnhappyFlows:
    """Test suite for unhappy flow scenarios (Part 5)"""
    
    def test_missing_parameters_agent(self):
        """Test missing parameters in agent endpoint"""
        # Empty message
        response = client.post("/agent/chat", json={"message": ""})
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()
        
        # Missing message field
        response = client.post("/agent/chat", json={})
        assert response.status_code == 422  # Validation error
    
    def test_missing_parameters_products(self):
        """Test missing parameters in products endpoint"""
        # Empty query
        response = client.get("/products/?query=")
        assert response.status_code == 400
        
        # Missing query parameter
        response = client.get("/products/")
        assert response.status_code == 422  # Validation error
    
    def test_missing_parameters_outlets(self):
        """Test missing parameters in outlets endpoint"""
        # Empty query
        response = client.get("/outlets/?query=")
        assert response.status_code == 400
        
        # Missing query parameter
        response = client.get("/outlets/")
        assert response.status_code == 422  # Validation error
    
    def test_malicious_sql_injection_outlets(self):
        """Test SQL injection attempts in outlets endpoint"""
        malicious_queries = [
            "'; DROP TABLE outlets; --",
            "1' OR '1'='1",
            "admin'; DELETE FROM outlets; --",
            "1 UNION SELECT * FROM outlets",
            "'; INSERT INTO outlets VALUES ('hack'); --"
        ]
        
        for query in malicious_queries:
            response = client.get(f"/outlets/?query={query}")
            assert response.status_code == 400
            assert "dangerous" in response.json()["detail"].lower()
    
    def test_malicious_xss_attempts(self):
        """Test XSS attempts in various endpoints"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert(1)'></iframe>"
        ]
        
        for payload in xss_payloads:
            # Test agent endpoint
            response = client.post("/agent/chat", json={"message": payload})
            # Should not crash, should handle gracefully
            assert response.status_code in [200, 400]
            
            # Test products endpoint
            response = client.get(f"/products/?query={payload}")
            assert response.status_code in [200, 400]
    
    def test_excessive_input_length(self):
        """Test handling of excessively long inputs"""
        long_input = "A" * 10000
        
        # Test agent endpoint
        response = client.post("/agent/chat", json={"message": long_input})
        assert response.status_code == 422  # Should fail validation
        
        # Test products endpoint
        response = client.get(f"/products/?query={long_input}")
        assert response.status_code == 422  # Should fail validation
    
    @patch('services.products_service.ZUSProductsService.search_products')
    def test_api_downtime_simulation_products(self, mock_search):
        """Test API downtime simulation for products service"""
        # Simulate service failure
        mock_search.side_effect = Exception("Service temporarily unavailable")
        
        response = client.get("/products/?query=test")
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()
    
    @patch('services.outlets_service.ZUSOutletsService.search_outlets')
    def test_api_downtime_simulation_outlets(self, mock_search):
        """Test API downtime simulation for outlets service"""
        # Simulate service failure
        mock_search.side_effect = Exception("Database connection failed")
        
        response = client.get("/outlets/?query=test")
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()
    
    @patch('tools.planner.AgenticPlanner.plan_and_execute')
    def test_api_downtime_simulation_agent(self, mock_planner):
        """Test API downtime simulation for agent service"""
        # Simulate agent failure
        mock_planner.side_effect = Exception("LLM service unavailable")
        
        response = client.post("/agent/chat", json={"message": "test"})
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()
    
    def test_invalid_calculation_expressions(self):
        """Test invalid mathematical expressions"""
        invalid_expressions = [
            "import os",
            "__import__('os')",
            "exec('print(1)')",
            "eval('1+1')",
            "open('/etc/passwd')",
            "1/0",  # Division by zero
            "invalid_syntax++",
            "2**10000000"  # Extremely large calculation
        ]
        
        for expr in invalid_expressions:
            response = client.post("/agent/chat", json={"message": f"Calculate {expr}"})
            # Should not crash, should handle gracefully
            assert response.status_code == 200
            response_data = response.json()
            assert "error" in response_data["response"].lower() or "invalid" in response_data["response"].lower()
    
    def test_network_timeout_handling(self):
        """Test network timeout scenarios"""
        with patch('requests.get') as mock_get:
            # Simulate timeout
            mock_get.side_effect = Exception("Connection timeout")
            
            response = client.get("/products/?query=test")
            # Should handle gracefully, not crash
            assert response.status_code in [200, 500]
    
    def test_database_connection_failure(self):
        """Test database connection failure scenarios"""
        with patch('pymongo.MongoClient') as mock_client:
            # Simulate database connection failure
            mock_client.side_effect = Exception("Database connection failed")
            
            response = client.get("/products/?query=test")
            # Should handle gracefully
            assert response.status_code in [200, 500]
    
    def test_rate_limiting_simulation(self):
        """Test rate limiting behavior"""
        # Make multiple rapid requests
        responses = []
        for i in range(10):
            response = client.get(f"/products/?query=test{i}")
            responses.append(response.status_code)
        
        # All should be handled gracefully (no crashes)
        assert all(status in [200, 429, 500] for status in responses)
    
    def test_malformed_json_requests(self):
        """Test malformed JSON in POST requests"""
        malformed_requests = [
            '{"message": "test"',  # Missing closing brace
            '{"message": }',       # Missing value
            '{"message": "test", "extra": }',  # Invalid JSON
            '',                    # Empty body
            'not json at all'      # Not JSON
        ]
        
        for malformed in malformed_requests:
            response = client.post(
                "/agent/chat", 
                data=malformed, 
                headers={"Content-Type": "application/json"}
            )
            # Should return 422 (validation error) or 400 (bad request)
            assert response.status_code in [400, 422]
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters"""
        special_inputs = [
            "测试中文输入",  # Chinese characters
            "🚀🎉💻",        # Emojis
            "café naïve résumé",  # Accented characters
            "null\x00byte",   # Null byte
            "\n\r\t",         # Control characters
        ]
        
        for special_input in special_inputs:
            response = client.post("/agent/chat", json={"message": special_input})
            # Should handle gracefully
            assert response.status_code in [200, 400]