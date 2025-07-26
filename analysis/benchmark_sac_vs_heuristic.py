import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.sac_agent_model import SACAgent

# Setup logging
logging.basicConfig(
    filename='logs/benchmark_sac_vs_heuristic.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger()

def reward_function(cost, performance):
    return -0.7 * cost + 0.3 * performance

def heuristic_policy(norm_usage):
    if norm_usage > 0.75:
        return 'on_demand'
    elif norm_usage < 0.25:
        return 'spot'
    else:
        return 'on_demand'

def penalty_function(interruption_rate):
    return interruption_rate

def benchmark(model_path, aws_data_path, interruption_data_path):
    logger.info(f"Loading AWS usage data from {aws_data_path}")
    df = pd.read_csv(aws_data_path, parse_dates=['usage_start_time'])
    logger.info(f"AWS usage data size: {len(df)}")

    logger.info(f"Loading interruption data from {interruption_data_path}")
    interruption_df = pd.read_csv(interruption_data_path)
    if 'interrupted' not in interruption_df.columns:
        raise KeyError("Interruption data must contain 'interrupted' column")

    overall_interruption_rate = interruption_df['interrupted'].mean()
    logger.info(f"Overall interruption rate: {overall_interruption_rate:.4f}")

    # Normalize features
    df['norm_cost'] = (df['cost'] - df['cost'].min()) / (df['cost'].max() - df['cost'].min())
    df['norm_usage'] = (df['usage'] - df['usage'].min()) / (df['usage'].max() - df['usage'].min())
    df['hour'] = df['usage_start_time'].dt.hour
    df['day'] = df['usage_start_time'].dt.dayofweek

    # Load SAC agent
    state_size = 4
    agent = SACAgent(state_dim=state_size, action_dim=1)  # action_dim=1 for continuous action
    agent.load(model_path)
    logger.info("SAC agent model loaded")

    sac_results = []
    heuristic_results = []

    for _, row in df.iterrows():
        actual_cost = row['cost']
        norm_usage = row['norm_usage']
        hour = row['hour']
        day = row['day']

        state = np.array([
            row['norm_cost'],
            norm_usage,
            hour / 23.0,
            day / 6.0
        ], dtype=np.float32)

        # SAC agent decision (continuous action)
        action = agent.select_action(state, evaluate=True)
        sac_choice = 'on_demand' if action < 0.5 else 'spot'

        sac_perf = 100 if sac_choice == 'on_demand' else 80
        sac_reward = reward_function(actual_cost, sac_perf)
        sac_penalty = penalty_function(overall_interruption_rate) if sac_choice == 'spot' else 0

        sac_results.append({
            'timestamp': row['usage_start_time'],
            'service': row['service'],
            'cost_actual': actual_cost,
            'choice': sac_choice,
            'reward': sac_reward,
            'penalty': sac_penalty,
            'policy_type': 'SAC'
        })

        # Heuristic policy
        heuristic_choice = heuristic_policy(norm_usage)
        heuristic_perf = 100 if heuristic_choice == 'on_demand' else 80
        heuristic_reward = reward_function(actual_cost, heuristic_perf)
        heuristic_penalty = penalty_function(overall_interruption_rate) if heuristic_choice == 'spot' else 0

        heuristic_results.append({
            'timestamp': row['usage_start_time'],
            'service': row['service'],
            'cost_actual': actual_cost,
            'choice': heuristic_choice,
            'reward': heuristic_reward,
            'penalty': heuristic_penalty,
            'policy_type': 'Heuristic'
        })

    # Combine results
    full_df = pd.DataFrame(sac_results + heuristic_results)

    # Calculate averages
    avg_sac_reward = full_df[full_df['policy_type'] == 'SAC']['reward'].mean()
    avg_sac_penalty = full_df[full_df['policy_type'] == 'SAC']['penalty'].mean()

    avg_heuristic_reward = full_df[full_df['policy_type'] == 'Heuristic']['reward'].mean()
    avg_heuristic_penalty = full_df[full_df['policy_type'] == 'Heuristic']['penalty'].mean()

    # Logging & Printing
    logger.info(f"Avg SAC Reward: {avg_sac_reward:.4f}, Penalty: {avg_sac_penalty:.4f}")
    logger.info(f"Avg Heuristic Reward: {avg_heuristic_reward:.4f}, Penalty: {avg_heuristic_penalty:.4f}")

    print(f"[SAC] Avg Reward: {avg_sac_reward:.2f}, Avg Penalty: {avg_sac_penalty:.3f}")
    print(f"[Heuristic] Avg Reward: {avg_heuristic_reward:.2f}, Avg Penalty: {avg_heuristic_penalty:.3f}")

    # Plotting
    plot_comparison(full_df)

    # Save results
    save_results(full_df)

def plot_comparison(df):
    sns.set(style="whitegrid")
    
    # Reward plot
    plt.figure(figsize=(14, 6))
    sns.lineplot(x='timestamp', y='reward', hue='policy_type', data=df)
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.title('Reward Over Time: SAC vs Heuristic')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('visuals', exist_ok=True)
    plt.savefig('visuals/sac_vs_heuristic_reward.png')
    plt.close()
    
    # Penalty plot
    plt.figure(figsize=(14, 6))
    sns.lineplot(x='timestamp', y='penalty', hue='policy_type', data=df)
    plt.xlabel('Time')
    plt.ylabel('Interruption Penalty')
    plt.title('Interruption Penalty Over Time: SAC vs Heuristic')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('visuals/sac_vs_heuristic_penalty.png')
    plt.close()

def save_results(df):
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sac_vs_heuristic_comparison.csv', index=False)
    logger.info("Comparison results saved to CSV.")

# Paths
MODEL_PATH = 'models/sac_agent.pth'
AWS_DATA_PATH = 'data/aws_hybrid_usage.csv'
INTERRUPTION_DATA_PATH = 'data/aws_hybrid_spot_instances.csv'

benchmark(MODEL_PATH, AWS_DATA_PATH, INTERRUPTION_DATA_PATH)