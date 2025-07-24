# I'm using sythentic data here due to real data insufficiency. But we can use real data files here instead of the synthetic one & they'll work same.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from lifelines import KaplanMeierFitter
import os
import logging

# Path setup 
script_dir = os.path.dirname(os.path.abspath(__file__))   # Get directory where script is located
data_dir = os.path.join(script_dir, '..', 'data')         # data folder
visuals_dir = os.path.join(script_dir, '..', 'visuals')   # visuals folder
logs_dir = os.path.join(script_dir, '..', 'logs')         # logs folder

# Logging 
log_file = os.path.join(logs_dir, 'aws_eda_forecast.log')
logging.basicConfig(
    filename=log_file,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Starting EDA and Forecasting script.")

# Data paths
real_usage_path = os.path.join(data_dir, 'aws_cost_export.csv')
synthetic_usage_path = os.path.join(data_dir, 'aws_simulated_usage.csv')
synthetic_spot_path = os.path.join(data_dir, 'aws_synthetic_spot_instances.csv')


# Load data
def load_real_data(path):
    try:
        df = pd.read_csv(path, parse_dates=['date'])
        logging.info(f"Successfully loaded {path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        return None

def load_synthetic_usage(path):
    try:
        df = pd.read_csv(path, parse_dates=['usage_start_time'])
        logging.info(f"Successfully loaded {path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        return None

def load_spot_data(path):
    try:
        df = pd.read_csv(path)
        logging.info(f"Successfully loaded {path}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        return None


df_real = load_real_data(real_usage_path)
df_synth = load_synthetic_usage(synthetic_usage_path)
df_spot = load_spot_data(synthetic_spot_path)


def plot_cost_trend(df, date_col, title, filename):
    plt.figure(figsize=(12,6))
    sns.lineplot(data=df, x=date_col, y='cost', marker='o')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cost ($)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    logging.info(f"Created cost trend plot: {filename}")

def forecast_cost(df, date_col, title, filename):
    df_prophet = df[[date_col, 'cost']].rename(columns={date_col: 'ds', 'cost': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    ax = fig1.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + 0.05, box.width, box.height * 0.92])
    plt.suptitle(f"30 Day Forecast", fontsize=14) 
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    logging.info(f"Created 30 Day Forecast plot: {filename}")

    fig2 = model.plot_components(forecast)
    components_plot_file = f"{filename}_components.png"
    plt.savefig(components_plot_file)
    plt.show()
    logging.info(f"Created forecast coponents plot : {components_plot_file}")

# EDA & Forecasting 
"""if df_real is not None:
    plot_cost_trend(df_real, 'date', 'Real AWS Cost Trend', os.path.join(visuals_dir, 'real_cost_trend.png'))
    forecast_cost(df_real, 'date', 'Real AWS Cost', os.path.join(visuals_dir, 'real_cost_forecast.png'))"""

if df_synth is not None:
    plot_cost_trend(df_synth, 'usage_start_time', 'AWS Cost Trend', os.path.join(visuals_dir, 'cost_trend.png'))
    forecast_cost(df_synth, 'usage_start_time', 'AWS Cost', os.path.join(visuals_dir, 'cost_forecast.png'))

# Survival analysis
if df_spot is not None and not df_spot.empty:
    if 'duration_hours' in df_spot.columns and 'interrupted' in df_spot.columns:
        kmf = KaplanMeierFitter()
        T = df_spot['duration_hours']
        E = df_spot['interrupted']
        kmf.fit(T, event_observed=E)
        ax = kmf.plot_survival_function()
        ax.set_title('Survival Function of Spot Instances')
        ax.set_xlabel('Duration (hours)')
        ax.set_ylabel('Survival Probability')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(visuals_dir, 'spot_instance_survival.png'))
        plt.show()
        logging.info(f"Created survival analysis on : {synthetic_spot_path}")
    else:
        logging.warning("Spot instance data missing required columns.")
else:
    logging.warning("Spot instance data is empty or missing.")