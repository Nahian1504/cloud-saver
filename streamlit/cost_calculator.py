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
        current_spend = col1.number_input(
            "Current Monthly Spend ($)",
            min_value=1000.0,
            max_value=1000000.0,
            value=float(cost_data['cost'].sum()),
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
        
        fig, ax = plt.subplots()
        ax.plot(range(1, months+1), cumulative)
        ax.set_title("Cumulative Savings Projection")
        ax.set_xlabel("Months")
        ax.set_ylabel("Savings ($)")
        st.pyplot(fig)
        
        logger.info(f"Calculated savings: {monthly_savings} at {optimization_level} level")
        
    except Exception as e:
        logger.error(f"Savings calculator error: {str(e)}", exc_info=True)
        st.error("Error in savings calculation")

def setup_logging():
    """Configure logging for the module"""
    import os
    from datetime import datetime
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(log_dir, f'calculator_{datetime.now().strftime("%Y%m%d")}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )