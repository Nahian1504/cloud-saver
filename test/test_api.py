import pytest
import sys
import os
import numpy as np
import json
import logging
import torch
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.main import app, STATE_DIM, ACTION_DIM

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Test client fixture with mock SAC agent"""
    with patch("api.main.agent") as mock_agent:  # Changed from sac_agent to agent
        # Configure mock agent to match your actual implementation
        mock_agent.STATE_DIM = STATE_DIM
        mock_agent.ACTION_DIM = ACTION_DIM
        mock_agent.select_action.return_value = torch.tensor(
            [0.5]
        )  # Single action for ACTION_DIM=1

        with app.test_client() as client:
            yield client


def test_health_check(client):
    """Test health check endpoint"""
    logger.info("Testing health check endpoint")
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json["api_status"] == "running"  # Matches main.py
    assert isinstance(response.json["model_loaded"], bool)
    assert "python_version" in response.json
    assert "torch_version" in response.json


def test_predict_valid_request(client):
    """Test prediction with valid request"""
    logger.info("Testing prediction with valid request")

    test_state = list(np.random.randn(STATE_DIM))  # Use actual STATE_DIM
    response = client.post(
        "/predict",
        json={"state": test_state},  # Using json parameter instead of data
    )

    assert response.status_code == 200
    assert response.json["status"] == "success"
    assert len(response.json["predicted_action"]) == ACTION_DIM
    assert len(response.json["input_state"]) == STATE_DIM
    assert "timestamp" in response.json


def test_predict_missing_state(client):
    """Test prediction with missing state"""
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert response.json["status"] == "error"
    assert "'state' parameter is required" in response.json["message"]


def test_predict_invalid_dimensions(client):
    """Test prediction with invalid state dimensions"""
    # Test with wrong dimensions
    response = client.post("/predict", json={"state": [1, 2, 3]})  # Should be 4
    assert response.status_code == 400
    assert f"length {STATE_DIM}" in response.json["message"]

    # Test with non-array input
    response = client.post("/predict", json={"state": "invalid"})
    assert response.status_code == 400
    assert "1D array" in response.json["message"]


def test_logging_output(client, caplog):
    """Test that the API logs properly"""
    with caplog.at_level(logging.DEBUG):
        client.get("/health")
        assert "Health check requested" in caplog.text

    test_state = list(np.random.randn(STATE_DIM))
    with caplog.at_level(logging.INFO):
        client.post("/predict", json={"state": test_state})
        assert "Prediction successful" in caplog.text


def test_error_handling(client, caplog):
    """Test error logging"""
    with caplog.at_level(logging.ERROR):
        # Trigger an error with invalid input
        client.post("/predict", json={"state": "invalid"})
        assert "Prediction error" in caplog.text
