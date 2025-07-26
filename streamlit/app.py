import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from utils import load_data, validate_data
from cost_calculator import calculate_savings
import os
from datetime import datetime

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'streamlit_app.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    st.set_page_config(
        page_title="Cloud Cost Optimizer",
        page_icon="💰",
        layout="wide"
    )
    
    logger.info("Initializing Streamlit dashboard")
    
    try:
        # Load data
        cost_data = load_data("data/aws_hybrid_usage.csv")
        spot_data = load_data("data/aws_hybrid_spot_instances.csv")
        
        if not validate_data(cost_data) or not validate_data(spot_data):
            st.error("Invalid data format. Please check the logs.")
            logger.error("Data validation failed")
            return
        
        # Dashboard Header
        st.title("AWS Cloud Cost Optimizer Dashboard")
        st.markdown("""
        Analyze and optimize your AWS spending with predictive analytics
        """)
        
        # Main Tabs
        tab1, tab2, tab3 = st.tabs(["Cost Analysis", "Spot Instance Optimizer", "Savings Calculator"])
        
        with tab1:
            render_cost_analysis(cost_data)
            
        with tab2:
            render_spot_optimizer(spot_data)
            
        with tab3:
            calculate_savings(cost_data)
            
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}", exc_info=True)
        st.error("An error occurred. Please check the logs.")
        st.stop()

def render_cost_analysis(data):
    """Cost trend visualization"""
    logger.info("Rendering cost analysis")
    
    st.header("Cost Trend Analysis")
    
    # Date range selector
    min_date = data['usage_start_time'].min().date()
    max_date = data['usage_start_time'].max().date()
    date_range = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        filtered = data[
            (data['usage_start_time'].dt.date >= date_range[0]) &
            (data['usage_start_time'].dt.date <= date_range[1])
        ]
        
        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        if 'data_type' in filtered.columns:
            for dtype, color in [('real', 'blue'), ('synthetic', 'orange')]:
                subset = filtered[filtered['data_type'] == dtype]
                if not subset.empty:
                    ax.plot(subset['usage_start_time'], subset['cost'], label=dtype.capitalize(), color=color)
            plt.legend()
        else:
            ax.plot(filtered['usage_start_time'], filtered['cost'])
            
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
    
    if 'duration_hours' not in data.columns or 'interrupted' not in data.columns:
        st.error("Missing required columns in spot instance data")
        logger.error("Missing columns in spot data")
        return
    
    # Survival analysis
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(data['duration_hours'], event_observed=data['interrupted'])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    kmf.plot_survival_function(ax=ax)
    ax.set_title("Spot Instance Survival Probability")
    ax.set_xlabel("Duration (hours)")
    ax.set_ylabel("Survival Probability")
    st.pyplot(fig)
    
    # Optimization recommendations
    st.subheader("Optimization Recommendations")
    if data['interrupted'].mean() > 0.3:
        st.warning("High interruption rate detected (>{:.0%})".format(0.3))
        st.markdown("""
        - Consider diversifying across instance types
        - Use Spot Blocks for critical workloads
        - Monitor Spot Instance Advisor for capacity trends
        """)
    else:
        st.success("Healthy spot instance performance")
        
    logger.info("Spot instance analysis rendered")

if __name__ == "__main__":
    main()