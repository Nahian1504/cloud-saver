import pandas as pd
import numpy as np
import random
import datetime
import os
import logging
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# --- Setup ---
log_file = "logs/generate_hybrid_aws_data.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('docs', exist_ok=True)

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

logger.info("Starting ENHANCED HYBRID data generation...")

# --- Load Real Data ---
real_data = pd.read_csv('data/aws_cost_export.csv')
real_data['date'] = pd.to_datetime(real_data['date'])
logger.info(f"Loaded {len(real_data)} real data points")

# --- Temporal Pattern Analysis with Fallback ---
def analyze_temporal_patterns(df):
    """Extracts trends and seasonality with robust small-data handling"""
    daily_costs = df.groupby('date')['cost'].sum().reset_index()
    daily_costs.set_index('date', inplace=True)
    
    if len(daily_costs) < 14:  # Insufficient for seasonal_decompose
        logger.warning(f"Insufficient data ({len(daily_costs)} points). Using fallback patterns.")
        base_value = daily_costs['cost'].mean() if not daily_costs.empty else 10.0
        
        # Create synthetic weekly pattern (Mon-Sun)
        seasonal_pattern = [0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 0.8]  # Typical AWS weekly fluctuation
        
        return {
            'trend': pd.Series([base_value], index=daily_costs.index),
            'seasonal': pd.Series(
                seasonal_pattern[:len(daily_costs)] if len(daily_costs) > 0 else [1.0],
                index=daily_costs.index
            ),
            'residual_std': daily_costs['cost'].std() * 0.3 if len(daily_costs) > 1 else 2.0
        }
    else:
        return seasonal_decompose(daily_costs, period=7)

# --- Enhanced Hybrid Generation ---
def generate_hybrid_data(real_df, num_days=30):
    """Generates hybrid dataset with robust small-data handling"""
    # 1. Handle parameter estimation
    if len(real_df) < 5:
        logger.warning("Very small real dataset - using industry benchmarks")
        cost_params = (0.3, 0, 10)  # Lognorm(s=0.3, loc=0, scale=10) for AWS
        usage_params = (2, 0, 5)    # Gamma(a=2, loc=0, scale=5) for usage
    else:
        try:
            cost_params = stats.lognorm.fit(real_df['cost'])
            usage_params = stats.gamma.fit(real_df['usage'])
        except:
            cost_params = (0.3, 0, 10)
            usage_params = (2, 0, 5)
    
    # 2. Get temporal patterns
    patterns = analyze_temporal_patterns(real_df)
    
    # 3. Generate synthetic dates
    last_real_date = real_df['date'].max() if not real_df.empty else datetime.date.today()
    synth_dates = pd.date_range(
        start=last_real_date + datetime.timedelta(days=1),
        periods=num_days
    )
    
    # 4. Create baseline trend
    last_trend = patterns['trend'].iloc[-1] if not patterns['trend'].empty else 10.0
    trend_values = np.linspace(last_trend, last_trend * 1.15, num_days)  # 15% monthly increase
    
    # 5. Generate synthetic data
    hybrid_data = []
    services = real_df['service'].unique() if not real_df.empty else ['AmazonEC2', 'AmazonS3']
    
    for i, date in enumerate(synth_dates):
        # Get seasonal component
        seasonal_idx = date.dayofweek % len(patterns['seasonal']) if not patterns['seasonal'].empty else 0
        seasonal_factor = patterns['seasonal'].iloc[seasonal_idx] if not patterns['seasonal'].empty else 1.0
        
        # Combine factors
        day_factor = seasonal_factor * (trend_values[i] / last_trend)
        noise = np.random.normal(0, patterns['residual_std'] * 0.5)  # Reduced noise
        
        for service in services:
            # Generate base values
            base_cost = stats.lognorm(*cost_params).rvs(1)[0]
            base_usage = stats.gamma(*usage_params).rvs(1)[0]
            
            # Apply adjustments
            cost = round(max(0.1, base_cost * day_factor + noise), 2)
            usage = round(base_usage * (0.9 + 0.2 * random.random()), 2)  # ±10% variation
            
            hybrid_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'service': service,
                'cost': cost,
                'usage': usage,
                'data_type': 'synthetic'
            })
    
    # Combine with real data
    if not real_df.empty:
        real_df['data_type'] = 'real'
        return pd.concat([real_df, pd.DataFrame(hybrid_data)])
    else:
        return pd.DataFrame(hybrid_data)

# --- Generate Hybrid Dataset ---
hybrid_df = generate_hybrid_data(real_data)
hybrid_df['usage_start_time'] = pd.to_datetime(hybrid_df['date'])
hybrid_df.drop(columns=['date'], inplace=True)

# Save Hybrid Data
output_path = os.path.join('data', 'aws_hybrid_usage.csv')
hybrid_df.to_csv(output_path, index=False)

logger.info(f"Saved hybrid AWS data to: {output_path}")
logger.info(f"Breakdown:\n{hybrid_df['data_type'].value_counts()}")

# --- Enhanced Spot Instance Data ---
def enhance_spot_data(real_spot_file=None):
    """Generates spot instance data with robust defaults"""
    # Default parameters
    duration_params = (0, 24)  # Exponential(24)
    interrupt_prob = 0.3
    
    # Try loading real data if available
    if real_spot_file and os.path.exists(real_spot_file):
        try:
            real_spot = pd.read_csv(real_spot_file)
            if len(real_spot) >= 5:  # Only use if sufficient data
                duration_params = stats.expon.fit(real_spot['duration_hours'])
                interrupt_prob = real_spot['interrupted'].mean()
        except:
            logger.warning("Failed to process real spot data - using defaults")
    
    # Generate data
    instance_ids = [f'i-{random.randint(1000,9999)}' for _ in range(50)]
    duration_hours = stats.expon(*duration_params).rvs(50)
    interrupted = np.random.binomial(1, p=interrupt_prob, size=50)
    
    # Add realistic time patterns
    start_times = pd.to_datetime('2024-01-01') + pd.to_timedelta(
        np.random.exponential(scale=48, size=50), unit='h'
    )
    
    spot_df = pd.DataFrame({
        'instance_id': instance_ids,
        'launch_time': start_times,
        'duration_hours': duration_hours,
        'interrupted': interrupted,
        'data_type': 'synthetic'
    })
    
    return spot_df

spot_df = enhance_spot_data()  # Pass path if real spot data exists
output_path_spot = os.path.join('data', 'aws_hybrid_spot_instances.csv')
spot_df.to_csv(output_path_spot, index=False)

logger.info(f"Saved enhanced spot instance data to: {output_path_spot}")

# --- Validation Report ---
def generate_validation_report(hybrid_df):
    """Creates validation output even with small data"""
    plt.figure(figsize=(15, 10))
    
    # 1. Cost Distribution
    plt.subplot(2, 2, 1)
    if 'data_type' in hybrid_df.columns:
        real = hybrid_df[hybrid_df['data_type'] == 'real']['cost'] if 'real' in hybrid_df['data_type'].values else pd.Series()
        synth = hybrid_df[hybrid_df['data_type'] == 'synthetic']['cost']
    else:
        real = pd.Series()
        synth = hybrid_df['cost']
    
    if not real.empty:
        plt.hist(real, alpha=0.5, label='Real', bins=min(20, len(real)), density=True)
    plt.hist(synth, alpha=0.5, label='Synthetic', bins=20, density=True)
    plt.title('Cost Distribution')
    plt.legend()
    
    # 2. Temporal Patterns
    plt.subplot(2, 2, 2)
    hybrid_df['usage_start_time'] = pd.to_datetime(hybrid_df['usage_start_time'])
    daily_costs = hybrid_df.groupby(['usage_start_time'])['cost'].mean()
    daily_costs.plot(ax=plt.gca())
    plt.title('Daily Cost Trend')
    
    # 3. Service Distribution
    plt.subplot(2, 2, 3)
    if 'service' in hybrid_df.columns:
        hybrid_df['service'].value_counts().plot(kind='bar')
    plt.title('Service Distribution')
    
    # 4. QQ Plot (if enough real data)
    plt.subplot(2, 2, 4)
    if not real.empty and len(real) > 5:
        stats.probplot(real, dist=stats.lognorm(*stats.lognorm.fit(real)), plot=plt)
        plt.title('QQ Plot: Real Data')
    else:
        plt.text(0.5, 0.5, 'Insufficient real data for QQ plot', ha='center')
    
    plt.tight_layout()
    plt.savefig('docs/hybrid_validation.png')
    logger.info("Generated validation report at docs/hybrid_validation.png")

generate_validation_report(hybrid_df)
logger.info("Hybrid data generation and validation completed successfully.")
