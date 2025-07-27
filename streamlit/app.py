import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from utils import load_data, validate_data
from cost_calculator import calculate_savings
from pathlib import Path
import os
from datetime import datetime
import sys

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    handlers=[
        logging.FileHandler(
            os.path.join(log_dir, f'streamlit_{datetime.now().strftime("%Y%m%d")}.log')
        ),
        logging.StreamHandler(sys.stdout),
    ],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_data_file(filename):
    """Search for data file in common locations"""
    search_paths = [
        Path("data") / filename,  # Relative to project root
        Path(__file__).parent.parent / "data" / filename,  # From streamlit/ folder
        Path.home() / "cloud-saver" / "data" / filename,  # User home directory
    ]

    for path in search_paths:
        if path.exists():
            return path
    return None


def main():
    st.set_page_config(page_title="Cloud Cost Optimizer", page_icon="💰", layout="wide")

    logger.info("Initializing Streamlit dashboard")

    try:
        # Data file discovery with fallback
        cost_data_path = find_data_file("aws_hybrid_usage.csv")
        spot_data_path = find_data_file("aws_hybrid_spot_instances.csv")

        # Debug panel
        with st.sidebar.expander("Debug Information"):
            st.write("Current directory:", Path.cwd())
            st.write("Cost data path:", cost_data_path)
            st.write("Spot data path:", spot_data_path)
            st.write(
                "File exists:", cost_data_path.exists() if cost_data_path else False
            )

            if st.button("Refresh Data"):
                st.experimental_rerun()

        # Handle missing files
        if not cost_data_path or not cost_data_path.exists():
            missing_files = []
            if not cost_data_path or not cost_data_path.exists():
                missing_files.append("aws_hybrid_usage.csv")
            if not spot_data_path or not spot_data_path.exists():
                missing_files.append("aws_hybrid_spot_instances.csv")

            st.error(f"Missing data files: {', '.join(missing_files)}")
            st.info(
                f"Search locations:\n1. {Path('data').absolute()}\n2. {Path(__file__).parent.parent / 'data'}"
            )

            if st.checkbox("Generate sample data (temporary)"):
                generate_sample_data()
                st.experimental_rerun()
            return

        # Load data with validation
        cost_data = load_data(cost_data_path)
        spot_data = load_data(spot_data_path)

        if not validate_data(cost_data) or not validate_data(spot_data):
            error_msg = "Invalid data format. Please check the logs."
            st.error(error_msg)
            logger.error(
                f"Data validation failed. Cost data columns: {cost_data.columns}, Spot data columns: {spot_data.columns}"
            )

            with st.expander("Data Validation Details"):
                st.write("Cost Data:", cost_data.head())
                st.write("Spot Data:", spot_data.head())
            return

        # Dashboard Header
        st.title("AWS Cloud Cost Optimizer Dashboard")
        st.markdown(
            """
        Analyze and optimize your AWS spending with predictive analytics
        """
        )

        # Main Tabs
        tab1, tab2, tab3 = st.tabs(
            ["Cost Analysis", "Spot Instance Optimizer", "Savings Calculator"]
        )

        with tab1:
            render_cost_analysis(cost_data)

        with tab2:
            render_spot_optimizer(spot_data)

        with tab3:
            calculate_savings(cost_data)

    except Exception as e:
        logger.critical(f"Critical dashboard error: {str(e)}", exc_info=True)
        st.error("A critical error occurred. Please check the logs.")
        st.exception(e)
        st.stop()


def generate_sample_data():
    """Create sample data files for testing"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Sample cost data
    dates = pd.date_range(end=datetime.now(), periods=30)
    cost_data = pd.DataFrame(
        {
            "usage_start_time": dates,
            "cost": np.random.uniform(100, 500, 30),
            "service": np.random.choice(["EC2", "S3", "RDS"], 30),
        }
    )
    cost_data.to_csv(data_dir / "aws_hybrid_usage.csv", index=False)

    # Sample spot data
    spot_data = pd.DataFrame(
        {
            "instance_type": np.random.choice(
                ["m5.large", "c5.xlarge", "r5.2xlarge"], 50
            ),
            "duration_hours": np.random.exponential(10, 50),
            "interrupted": np.random.binomial(1, 0.2, 50),
        }
    )
    spot_data.to_csv(data_dir / "aws_hybrid_spot_instances.csv", index=False)

    st.success("Generated sample data in data/ directory")


def render_cost_analysis(data):
    """Cost trend visualization"""
    logger.info("Rendering cost analysis")

    st.header("Cost Trend Analysis")

    with st.expander("Data Summary"):
        st.write(
            f"Data from {data['usage_start_time'].min().date()} to {data['usage_start_time'].max().date()}"
        )
        st.write(data.describe())

    # Date range selector
    min_date = data["usage_start_time"].min().date()
    max_date = data["usage_start_time"].max().date()
    date_range = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if len(date_range) == 2:
        filtered = data[
            (data["usage_start_time"].dt.date >= date_range[0])
            & (data["usage_start_time"].dt.date <= date_range[1])
        ]

        if filtered.empty:
            st.warning("No data in selected date range")
            return

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        if "service" in filtered.columns:
            for service in filtered["service"].unique():
                subset = filtered[filtered["service"] == service]
                ax.plot(subset["usage_start_time"], subset["cost"], label=service)
            plt.legend()
        else:
            ax.plot(filtered["usage_start_time"], filtered["cost"])

        ax.set_title("Daily Cost Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cost ($)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cost", f"${filtered['cost'].sum():,.2f}")
        col2.metric("Avg Daily Cost", f"${filtered['cost'].mean():,.2f}")
        col3.metric("Peak Cost", f"${filtered['cost'].max():,.2f}")

        logger.info("Cost analysis rendered successfully")
    else:
        st.warning("Please select a valid date range")
        logger.warning("Invalid date range selected")


def render_spot_optimizer(data):
    """Spot instance analysis"""
    logger.info("Rendering spot instance optimizer")

    st.header("Spot Instance Optimization")

    required_cols = ["duration_hours", "interrupted"]
    if not all(col in data.columns for col in required_cols):
        st.error(f"Missing required columns: {set(required_cols) - set(data.columns)}")
        logger.error(f"Missing columns in spot data: {data.columns}")
        return

    try:
        from lifelines import KaplanMeierFitter

        kmf = KaplanMeierFitter()
        kmf.fit(data["duration_hours"], event_observed=data["interrupted"])

        fig, ax = plt.subplots(figsize=(10, 5))
        kmf.plot_survival_function(ax=ax)
        ax.set_title("Spot Instance Survival Probability")
        ax.set_xlabel("Duration (hours)")
        ax.set_ylabel("Survival Probability")
        st.pyplot(fig)

        # Optimization recommendations
        interruption_rate = data["interrupted"].mean()
        st.subheader("Optimization Recommendations")
        if interruption_rate > 0.3:
            st.warning(f"High interruption rate detected ({interruption_rate:.0%})")
            st.markdown(
                """
            - Consider diversifying across instance types
            - Use Spot Blocks for critical workloads
            - Monitor Spot Instance Advisor for capacity trends
            """
            )
        else:
            st.success(
                f"Healthy spot instance performance (interruption rate: {interruption_rate:.0%})"
            )

        logger.info("Spot instance analysis rendered")
    except ImportError:
        st.error("Please install lifelines: pip install lifelines")
    except Exception as e:
        logger.error(f"Spot analysis failed: {str(e)}", exc_info=True)
        st.error("Error in spot instance analysis")


if __name__ == "__main__":
    main()
