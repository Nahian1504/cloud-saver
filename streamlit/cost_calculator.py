from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime
import os
from utils import validate_data


# Configure logging
def setup_logging():
    """Configure logging for the module"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                filename=os.path.join(
                    log_dir, f'calculator_{datetime.now().strftime("%Y%m%d")}.log'
                ),
                mode="a",
            ),
            logging.StreamHandler(sys.stdout),
        ],
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


setup_logging()
logger = logging.getLogger(__name__)


def calculate_savings(cost_data):
    """Interactive savings calculator"""
    try:
        logger.info("Initializing savings calculator")

        # Validate input data
        if not isinstance(cost_data, pd.DataFrame):
            raise ValueError("cost_data must be a pandas DataFrame")

        if not validate_data(cost_data):
            error_msg = "Invalid cost data provided"
            logger.error(error_msg)
            st.error(error_msg)
            return None

        st.header("Savings Calculator")

        # User inputs with safe defaults
        col1, col2 = st.columns(2)

        try:
            default_spend = max(1000.0, float(cost_data["cost"].sum()))
        except (KeyError, TypeError, ValueError) as e:
            default_spend = 1000.0
            logger.warning(f"Using fallback default spend: {e}")

        current_spend = col1.number_input(
            "Current Monthly Spend ($)",
            min_value=1000.0,
            max_value=1000000.0,
            value=default_spend,
            step=1000.0,
        )

        optimization_level = col2.select_slider(
            "Optimization Level", options=["Low", "Medium", "High"], value="Medium"
        )

        # Calculate savings
        savings_rates = {"Low": 0.15, "Medium": 0.25, "High": 0.35}
        savings_rate = savings_rates.get(optimization_level, 0.25)
        monthly_savings = current_spend * savings_rate

        # Display results
        st.metric("Estimated Monthly Savings", f"${monthly_savings:,.2f}")

        # Savings timeline visualization
        try:
            months = st.slider("Projection Period (months)", 1, 36, 12)
            time_period = np.arange(1, months + 1)
            cumulative = np.cumsum([monthly_savings] * months)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_period, cumulative, "b-", linewidth=2.5)
            ax.set_title("Cumulative Savings Projection", pad=20)
            ax.set_xlabel("Months", labelpad=10)
            ax.set_ylabel("Savings ($)", labelpad=10)
            ax.grid(True, linestyle="--", alpha=0.7)
            st.pyplot(fig)

            logger.info(
                f"Successfully calculated savings: ${monthly_savings:,.2f}/month at {optimization_level} level"
            )
            return monthly_savings

        except Exception as plot_error:
            logger.error(f"Plotting error: {str(plot_error)}", exc_info=True)
            st.warning("Could not generate savings projection")
            return monthly_savings  # Still return value even if plot fails

    except Exception as e:
        error_msg = f"Calculator error: {str(e)}"
        logger.critical(error_msg, exc_info=True)
        st.error("A critical error occurred. Please check the logs.")
        return None


# Example test function
def test_calculator():
    """Test the calculator with sample data"""
    test_data = pd.DataFrame({"cost": [1500.0, 2000.0, 1800.0]})
    return calculate_savings(test_data)


if __name__ == "__main__":
    st.set_page_config(page_title="Savings Calculator")
    result = test_calculator()
    if result is not None:
        st.success("Calculation completed successfully!")
