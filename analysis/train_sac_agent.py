import pandas as pd
import numpy as np
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.sac_agent_model import SACAgent  # Your SAC implementation

logging.basicConfig(
    filename='logs/train_sac_agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger()

# Reward function (same)
def reward_function(cost, performance):
    return -0.7 * cost + 0.3 * performance

# Feature extraction for RL agent
def get_state_features(row):
    cost = row['cost_norm']
    usage = row['usage_norm']
    hour = row['hour'] / 23.0
    day = row['day'] / 6.0
    return np.array([cost, usage, hour, day], dtype=np.float32)

# Tunable cost-usage weighted heuristic (unchanged)
def heuristic_policy(row, cost_weight=0.7, usage_weight=0.3, threshold=0.5):
    cost_score = 1 - row['cost_norm']
    usage_score = row['usage_norm']
    decision_score = cost_weight * cost_score + usage_weight * usage_score
    return 1 if decision_score > threshold else 0  # 1: spot, 0: on_demand

def train_agent(data_path, model_path, num_epochs=50):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, parse_dates=['usage_start_time'])
    df = df.sort_values('usage_start_time')

    # Normalize columns
    df['cost_norm'] = (df['cost'] - df['cost'].min()) / (df['cost'].max() - df['cost'].min())
    df['usage_norm'] = (df['usage'] - df['usage'].min()) / (df['usage'].max() - df['usage'].min())
    df['hour'] = df['usage_start_time'].dt.hour
    df['day'] = df['usage_start_time'].dt.dayofweek

    state_size = 4  # cost_norm, usage_norm, hour, day
    agent = SACAgent(state_dim=state_size, action_dim=1)  # action_dim=1 for continuous scalar

    rl_total_reward = 0
    heuristic_total_reward = 0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs} started")
        for idx, row in df.iterrows():
            state = get_state_features(row)

            # --- SAC agent decision ---
            action = agent.select_action(state)  # action is continuous scalar, e.g., in [0,1]

            # Interpret action for cost/performance:
            # For example: action < 0.5 means on_demand (0), else spot (1)
            rl_action = 0 if action < 0.5 else 1

            rl_cost = row['cost']
            rl_perf = 100 if rl_action == 0 else 80
            rl_reward = reward_function(rl_cost, rl_perf)
            rl_total_reward += rl_reward

            # --- Heuristic decision ---
            h_action = heuristic_policy(row)
            h_cost = row['cost']
            h_perf = 100 if h_action == 0 else 80
            h_reward = reward_function(h_cost, h_perf)
            heuristic_total_reward += h_reward

            # Train agent
            next_state = state  # assuming no state change in this simplified example
            done = False  # no episode end in this example

            agent.remember(state, action, rl_reward, next_state, done)
            agent.update()

        logger.info(f"Epoch {epoch+1} complete.")
        logger.info(f"Total RL Reward (so far): {rl_total_reward:.2f}")
        logger.info(f"Total Heuristic Reward (so far): {heuristic_total_reward:.2f}")

    agent.save(model_path)
    logger.info(f"Training complete. Model saved to {model_path}")
    logger.info(f"Final RL Total Reward: {rl_total_reward:.2f}")
    logger.info(f"Final Heuristic Total Reward: {heuristic_total_reward:.2f}")

    return model_path


# Paths
DATA_PATH = 'data/aws_synthetic_usage.csv'
MODEL_PATH = 'models/sac_agent.pth'


train_agent(DATA_PATH, MODEL_PATH, num_epochs=60)