import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api.main import app
import numpy as np
import json
import logging
from unittest.mock import patch

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def client():
    """Test client fixture with mock SAC agent"""
    with patch('api.main.agent') as mock_agent:
        # Configure mock agent
        mock_agent.STATE_DIM = 10
        mock_agent.ACTION_DIM = 2
        mock_agent.select_action.return_value = np.array([0.5, -0.3])
        
        with app.test_client() as client:
            yield client

def test_health_check(client):
    """Test health check endpoint"""
    logger.info("Testing health check endpoint")
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'
    assert 'model_loaded' in response.json
    assert 'device' in response.json

def test_predict_valid_request(client):
    """Test prediction with valid request"""
    logger.info("Testing prediction with valid request")
    
    test_state = list(np.random.randn(10))  # Match STATE_DIM
    response = client.post(
        '/predict',
        data=json.dumps({'state': test_state}),
        content_type='application/json'
    )
    
    logger.debug(f"Response: {response.json}")
    assert response.status_code == 200
    assert 'action' in response.json
    assert isinstance(response.json['action'], list)
    assert len(response.json['action']) == 2  # Match ACTION_DIM

def test_predict_missing_state(client):
    """Test prediction with missing state"""
    logger.info("Testing prediction with missing state")
    response = client.post('/predict', json={})
    assert response.status_code == 400
    assert 'error' in response.json
    assert 'state' in response.json['error']

def test_predict_invalid_dimensions(client):
    """Test prediction with invalid state dimensions"""
    logger.info("Testing prediction with invalid dimensions")
    
    # Test with too few dimensions
    response = client.post('/predict', json={'state': [1, 2, 3]})
    assert response.status_code == 400
    assert 'dimensions' in response.json['error']
    
    # Test with non-array input
    response = client.post('/predict', json={'state': "invalid"})
    assert response.status_code == 400
    assert 'array' in response.json['error']

def test_model_info(client):
    """Test model info endpoint"""
    logger.info("Testing model info endpoint")
    response = client.get('/model_info')
    assert response.status_code == 200
    assert response.json['model_type'] == 'SAC'
    assert response.json['state_dim'] == 10
    assert response.json['action_dim'] == 2
    assert 'policy_parameters' in response.json

def test_logging_output(client, caplog):
    """Test that the API logs properly"""
    with caplog.at_level(logging.INFO):
        client.get('/health')
        assert "Health check requested" in caplog.text

    test_state = list(np.random.randn(10))
    with caplog.at_level(logging.INFO):
        client.post('/predict', json={'state': test_state})
        assert "Received prediction request" in caplog.text
        assert "Prediction successful" in caplog.text

def test_error_handling(client, caplog):
    """Test error logging"""
    with caplog.at_level(logging.ERROR):
        # Trigger an error with invalid input
        client.post('/predict', json={'state': "invalid"})
        assert "Invalid state format" in caplog.text