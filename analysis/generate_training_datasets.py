import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_aws_data(num_records=10000, start_date='2024-01-01'):
    np.random.seed(42)
    random.seed(42)
    
    services = ['EC2', 'S3', 'Lambda', 'RDS', 'DynamoDB']
    data = []

    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')

    for i in range(num_records):
        service = random.choice(services)
        
        # Simulate usage hours [0, 24)
        usage_start_time = start_datetime + timedelta(hours=i % 24, days=i // 24)
        
        # Cost: EC2 > RDS > Lambda > DynamoDB > S3 (just an example scale)
        base_cost = {
            'EC2': 0.20,
            'RDS': 0.15,
            'Lambda': 0.05,
            'DynamoDB': 0.03,
            'S3': 0.01
        }[service]
        
        # Usage: normalized random usage between 0 and 1 scaled to hours (0-24)
        usage = np.clip(np.random.normal(loc=0.5, scale=0.2), 0, 1)
        
        # Cost fluctuates with usage + some noise
        cost = base_cost * usage * 24 * (1 + np.random.normal(0, 0.05))
        
        data.append({
            'service': service,
            'cost': round(cost, 4),
            'usage': round(usage, 4),
            'usage_start_time': usage_start_time
        })
    
    df = pd.DataFrame(data)
    return df

def generate_synthetic_interruption_data(services, num_records=5000):
    np.random.seed(42)
    random.seed(42)
    data = []
    for _ in range(num_records):
        service = random.choice(services)
        # Simulate interruption probability for spot instances per service
        # e.g. EC2 spots have higher interruption than S3
        interruption_prob = {
            'EC2': 0.10,
            'RDS': 0.05,
            'Lambda': 0.02,
            'DynamoDB': 0.01,
            'S3': 0.005
        }[service]
        interrupted = np.random.rand() < interruption_prob
        duration_hours = np.random.uniform(1, 24)
        instance_id = f'{service[:2].upper()}-{np.random.randint(10000,99999)}'
        data.append({
            'instance_id': instance_id,
            'duration_hours': round(duration_hours, 2),
            'interrupted': int(interrupted),
            'service': service  # add service to link with usage data
        })
    df = pd.DataFrame(data)
    return df


usage_df = generate_synthetic_aws_data()
usage_df.to_csv('data/aws_synthetic_usage.csv', index=False)

interruption_df = generate_synthetic_interruption_data(usage_df['service'].unique())
interruption_df.to_csv('data/aws_synthetic_interruption.csv', index=False)