import pandas as pd
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load and validate data file"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Data file {file_path} does not exist")
            
        df = pd.read_csv(file_path)
        
        # Common date column checks
        date_col = 'usage_start_time' if 'usage_start_time' in df.columns else 'date'
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            
        logger.info(f"Successfully loaded data from {file_path}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}", exc_info=True)
        raise

def validate_data(df):
    """Validate data structure"""
    try:
        if df is None or df.empty:
            logger.warning("Empty dataframe received")
            return False
            
        required_columns = {
            'aws_hybrid_usage.csv': ['usage_start_time', 'cost'],
            'aws_hybrid_spot_instances.csv': ['duration_hours', 'interrupted']
        }
        
        filename = [k for k in required_columns if k in str(df.attrs.get('filename', ''))]
        filename = filename[0] if filename else None
        
        if filename and not all(col in df.columns for col in required_columns[filename]):
            missing = [col for col in required_columns[filename] if col not in df.columns]
            logger.error(f"Missing required columns for {filename}: {missing}")
            return False
            
        # Check for negative costs
        if 'cost' in df.columns and (df['cost'] < 0).any():
            logger.warning("Negative costs detected in data")
            
        return True
        
    except Exception as e:
        logger.error(f"Validation error: {str(e)}", exc_info=True)
        return False

def setup_logging():
    """Configure logging for the module"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(log_dir, f'utils_{datetime.now().strftime("%Y%m%d")}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )