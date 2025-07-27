from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import logging
from utils import validate_data

logger = logging.getLogger(__name__)

def calculate_savings(cost_data):
    """Interactive savings calculator"""
    logger.info("Initializing savings calculator")
    
    try:
        if not validate_data(cost_data):
            st.error("Invalid cost data provided")
            logger.error("Invalid data passed to calculator")
            return
            
        st.header("Savings Calculator")
        
        # User inputs
        col1, col2 = st.columns(2)
        
        # Safely calculate default spend value
        default_spend = max(1000.0, float(cost_data['cost'].sum()))
        
        current_spend = col1.number_input(
            "Current Monthly Spend ($)",
            min_value=1000.0,
            max_value=1000000.0,
            value=default_spend,  # Ensures value ≥ min_value
            step=1000.0
        )
        
        optimization_level = col2.select_slider(
            "Optimization Level",
            options=['Low', 'Medium', 'High'],
            value='Medium'
        )
        
        # Calculate savings
        savings_rates = {'Low': 0.15, 'Medium': 0.25, 'High': 0.35}
        savings_rate = savings_rates[optimization_level]
        monthly_savings = current_spend * savings_rate
        
        # Display results
        st.metric("Estimated Monthly Savings", f"${monthly_savings:,.2f}")
        
        # Savings timeline
        months = st.slider("Projection Period (months)", 1, 36, 12)
        cumulative = np.cumsum([monthly_savings] * months)
        
        # Plot with improved formatting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, months+1), cumulative, 'b-', linewidth=2.5)
        ax.set_title("Cumulative Savings Projection", pad=20)
        ax.set_xlabel("Months", labelpad=10)
        ax.set_ylabel("Savings ($)", labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_facecolor('#f5f5f5')
        fig.patch.set_facecolor('#ffffff')
        st.pyplot(fig)
        
        logger.info(f"Calculated savings: {monthly_savings} at {optimization_level} level")
        
    except Exception as e:
        logger.error(f"Savings calculator error: {str(e)}", exc_info=True)
        st.error(f"Error in savings calculation: {str(e)}")

def setup_logging():
    """Configure logging for the module"""
    import os
    from datetime import datetime
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(log_dir, f'calculator_{datetime.now().strftime("%Y%m%d")}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'  # Append to log file instead of overwriting
    )

# Example usage (for testing)
if __name__ == "__main__":
    setup_logging()
    
    # Sample test data
    test_data = pd.DataFrame({
        'cost': [1500.0, 2000.0, 1800.0]
    })
    
    calculate_savings(test_data)