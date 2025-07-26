import pytest
import pandas as pd
import numpy as np
import logging
import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.sac_agent_model import SACAgent  # Changed from DQNAgent to SACAgent

# Setup test logging
logging.basicConfig(
    filename='logs/test_sac_agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger()

TEST_DATA_PATH = 'data/aws_hybrid_usage.csv'
MODEL_PATH = 'models/sac_agent.pth'  

@pytest.fixture(scope="module")
def test_data():
    """Load test dataset once per test module."""
    logger.info(f"Loading test data from {TEST_DATA_PATH}")
    df = pd.read_csv(TEST_DATA_PATH, parse_dates=['usage_start_time'])
    logger.info(f"Test data loaded with {len(df)} rows")
    return df

@pytest.fixture(scope="module")
def agent():
    """Load trained SACAgent once per test module."""
    # SAC agent with same state size but action_dim=1 for continuous action
    agent = SACAgent(state_dim=4, action_dim=1)
    logger.info(f"Loading SAC agent model from {MODEL_PATH}")
    agent.load(MODEL_PATH)
    return agent

def test_test_data_integrity(test_data):
    """Basic checks on test data."""
    logger.info(f"Test data columns: {test_data.columns.tolist()}")
    assert 'service' in test_data.columns
    assert 'cost' in test_data.columns
    assert 'usage_start_time' in test_data.columns
    assert len(test_data) > 0
    logger.info("Test data integrity check passed.")

def test_agent_select_action_method(agent, test_data):
    """Test that agent.select_action(state) returns valid continuous actions."""
    for _, row in test_data.head(10).iterrows():  # limit to 10 for speed
        state = np.array([
            (row['cost'] - test_data['cost'].min()) / (test_data['cost'].max() - test_data['cost'].min()),
            (row['usage'] - test_data['usage'].min()) / (test_data['usage'].max() - test_data['usage'].min()),
            row['usage_start_time'].hour / 23.0,
            row['usage_start_time'].dayofweek / 6.0
        ], dtype=np.float32)
        
        # Test both evaluation and exploration modes
        eval_action = agent.select_action(state, evaluate=True)
        explore_action = agent.select_action(state, evaluate=False)
        
        logger.info(f"Evaluation action: {eval_action}, Exploration action: {explore_action}")
        
        # Check actions are within expected range (-1 to 1 due to tanh)
        assert -1 <= eval_action <= 1, "Evaluation action out of bounds"
        assert -1 <= explore_action <= 1, "Exploration action out of bounds"

def test_model_parameters_loaded(agent):
    """Verify that all SAC networks have parameters loaded."""
    # Check policy network
    policy_params = list(agent.policy.parameters())
    policy_all_zeros = all(torch.equal(p.data, torch.zeros_like(p.data)) for p in policy_params)
    assert not policy_all_zeros, "Policy network parameters appear uninitialized"
    
    # Check Q networks
    q1_params = list(agent.q1.parameters())
    q1_all_zeros = all(torch.equal(p.data, torch.zeros_like(p.data)) for p in q1_params)
    assert not q1_all_zeros, "Q1 network parameters appear uninitialized"
    
    q2_params = list(agent.q2.parameters())
    q2_all_zeros = all(torch.equal(p.data, torch.zeros_like(p.data)) for p in q2_params)
    assert not q2_all_zeros, "Q2 network parameters appear uninitialized"

def test_target_network_updates(agent):
    """Verify target networks are different from main networks."""
    # Compare small samples of parameters
    for target_param, param in zip(agent.q1_target.parameters(), agent.q1.parameters()):
        assert not torch.equal(target_param.data, param.data), "Target network matches main network"
        break  # Just check first parameter

    for target_param, param in zip(agent.q2_target.parameters(), agent.q2.parameters()):
        assert not torch.equal(target_param.data, param.data), "Target network matches main network"
        break  # Just check first parameter

def test_replay_buffer_operation(agent):
    """Test basic replay buffer functionality."""
    # Add some dummy experiences
    dummy_state = np.random.randn(4)
    dummy_action = np.random.randn(1)
    dummy_reward = 1.0
    dummy_next_state = np.random.randn(4)
    dummy_done = False
    
    agent.remember(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)
    assert len(agent.memory) > 0, "Replay buffer should not be empty after adding experience"
    
    # Test update doesn't crash
    agent.update()
    logger.info("SAC update completed successfully")

if __name__ == "__main__":
    pytest.main()