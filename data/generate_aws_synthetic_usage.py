import pandas as pd
import numpy as np
import random
import datetime
import os
import logging



log_file = "logs/generate_aws_synthetic_usage.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Seed Initialization 
random.seed(42)
np.random.seed(42)

logger.info("Starting synthetic data generation...")

# Date and Service Setup 
today = datetime.date.today()
dates = [today - datetime.timedelta(days=i) for i in range(30)]
dates.reverse()

services = ['AmazonEC2', 'AmazonS3', 'AmazonRDS', 'AmazonCloudWatch', 'AmazonDynamoDB']

# Generate Usage Data 
data = []
for date in dates:
    for service in services:
        cost = round(random.uniform(0.5, 20.0), 2)
        usage = round(random.uniform(0.1, 10.0), 2)
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'service': service,
            'cost': cost,
            'usage': usage
        })

df = pd.DataFrame(data)
df['usage_start_time'] = pd.to_datetime(df['date'])
df.drop(columns=['date'], inplace=True)

# Save Usage 
output_path_usage = os.path.join('data', 'simulated_usage.csv')
df.to_csv(output_path_usage, index=False)

logger.info(f"Saved simulated AWS usage data to: {output_path_usage}")
logger.info(f"Total usage records: {len(df)}")

#  Spot Instance Survival Data
instance_ids = [f'i-{random.randint(1000,9999)}' for _ in range(50)]
duration_hours = np.random.exponential(scale=24, size=50)
interrupted = np.random.binomial(1, p=0.3, size=50)

spot_df = pd.DataFrame({
    'instance_id': instance_ids,
    'duration_hours': duration_hours,
    'interrupted': interrupted
})

output_path_spot = os.path.join('data', 'aws_synthetic_spot_instances.csv')
spot_df.to_csv(output_path_spot, index=False)

logger.info(f"Saved synthetic spot instance data to: {output_path_spot}")
logger.info(f"Total spot instances: {len(spot_df)}")

logger.info("Synthetic data generation completed successfully.")
