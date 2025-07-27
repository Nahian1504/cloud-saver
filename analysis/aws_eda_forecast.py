import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from lifelines import KaplanMeierFitter
import os
import logging
import numpy as np

# --- Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
visuals_dir = os.path.join(script_dir, "..", "visuals")
logs_dir = os.path.join(script_dir, "..", "logs")

os.makedirs(visuals_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

log_file = os.path.join(logs_dir, "aws_eda_forecast.log")
logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Starting EDA and Forecasting script.")


# --- Data Loading ---
def load_data(path, date_cols=None):
    try:
        df = pd.read_csv(path, parse_dates=date_cols)
        logging.info(f"Successfully loaded data: {path}")
        return df
    except Exception as e:
        logging.error(f"Error loading {path}: {str(e)}")
        return None


df_real = load_data(os.path.join(data_dir, "aws_cost_export.csv"), ["date"])
df_hybrid = load_data(
    os.path.join(data_dir, "aws_hybrid_usage.csv"), ["usage_start_time"]
)
df_spot = load_data(os.path.join(data_dir, "aws_hybrid_spot_instances.csv"))


# --- Visualization Functions ---
def plot_cost_trend(df, date_col, title, filename):
    """Fixed version with better legend handling"""
    fig, ax = plt.subplots(figsize=(12, 6))

    if "data_type" in df.columns:
        # Plot real and synthetic separately if available
        for dtype, color, label in [
            ("real", "blue", "Real"),
            ("synthetic", "orange", "Synthetic"),
        ]:
            if dtype in df["data_type"].values:
                subset = df[df["data_type"] == dtype]
                sns.lineplot(
                    data=subset,
                    x=date_col,
                    y="cost",
                    marker="o",
                    ax=ax,
                    color=color,
                    label=label,
                )
    else:
        sns.lineplot(data=df, x=date_col, y="cost", marker="o", ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cost ($)")
    plt.xticks(rotation=45)
    plt.grid(True)
    if "data_type" in df.columns:
        plt.legend()
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    logging.info(f"Created cost trend plot: {filename}")


def forecast_cost(df, date_col, title, filename):
    """Fixed Prophet forecasting with data validation"""
    try:
        # Data preparation
        df_clean = (
            df[[date_col, "cost"]]
            .dropna()
            .rename(columns={date_col: "ds", "cost": "y"})
        )

        # Remove zeros/infs that break Prophet
        df_clean = df_clean[(df_clean["y"] > 0) & (np.isfinite(df_clean["y"]))]

        if len(df_clean) < 2:
            raise ValueError("Insufficient valid data points for forecasting")

        # Model fitting
        model = Prophet(
            yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False
        )
        model.fit(df_clean)

        # Forecasting
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        # Save plots
        fig1 = model.plot(forecast)
        plt.title(f"{title} - 30 Day Forecast")
        plt.tight_layout()
        fig1.savefig(filename)
        plt.close(fig1)

        fig2 = model.plot_components(forecast)
        plt.tight_layout()
        components_file = filename.replace(".png", "_components.png")
        fig2.savefig(components_file)
        plt.close(fig2)

        logging.info(f"Successfully created forecast plots for {filename}")

    except Exception as e:
        logging.error(f"Forecast failed for {title}: {str(e)}")


def plot_spot_survival(df, filename):
    """Fixed survival analysis with data validation"""
    try:
        if not all(col in df.columns for col in ["duration_hours", "interrupted"]):
            raise ValueError("Missing required columns")

        kmf = KaplanMeierFitter()
        T = df["duration_hours"]
        E = df["interrupted"]

        if len(T) < 1:
            raise ValueError("No data available for survival analysis")

        kmf.fit(T, event_observed=E)

        fig, ax = plt.subplots()
        kmf.plot_survival_function(ax=ax)
        ax.set_title("Spot Instance Survival Function")
        ax.set_xlabel("Duration (hours)")
        ax.set_ylabel("Survival Probability")
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
        logging.info(f"Created survival plot: {filename}")

    except Exception as e:
        logging.error(f"Survival analysis failed: {str(e)}")


# --- Execute Analysis ---
if df_real is not None and len(df_real) >= 2:
    plot_cost_trend(
        df_real,
        "date",
        "Real AWS Cost Trend",
        os.path.join(visuals_dir, "real_cost_trend.png"),
    )
    forecast_cost(
        df_real,
        "date",
        "Real AWS Cost",
        os.path.join(visuals_dir, "real_cost_forecast.png"),
    )

if df_hybrid is not None:
    plot_cost_trend(
        df_hybrid,
        "usage_start_time",
        "Hybrid AWS Cost Trend",
        os.path.join(visuals_dir, "hybrid_cost_trend.png"),
    )
    forecast_cost(
        df_hybrid,
        "usage_start_time",
        "Hybrid AWS Cost",
        os.path.join(visuals_dir, "hybrid_cost_forecast.png"),
    )

if df_spot is not None:
    plot_spot_survival(df_spot, os.path.join(visuals_dir, "spot_survival.png"))

logging.info("Script completed successfully.")
