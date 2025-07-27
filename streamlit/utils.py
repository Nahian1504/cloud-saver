import pandas as pd
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load and validate data file with robust path handling
    
    Args:
        file_path (str or Path): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded and validated dataframe
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data validation fails
    """
    try:
        # Convert to Path object if not already
        path = Path(file_path) if not isinstance(file_path, Path) else file_path
        
        # Verify file exists with helpful debug info
        if not path.exists():
            available_files = []
            if path.parent.exists():
                available_files = [f.name for f in path.parent.glob('*') if f.is_file()]
            
            logger.error(
                f"File not found: {path.absolute()}\n"
                f"Directory contents: {available_files}"
            )
            raise FileNotFoundError(
                f"Data file not found at: {path.absolute()}\n"
                f"Available files: {available_files}"
            )
        
        # Load data with datetime parsing
        df = pd.read_csv(
            path,
            parse_dates=['usage_start_time', 'date'],
            infer_datetime_format=True,
            dayfirst=False,
            on_bad_lines='warn'
        )
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Date column handling
        date_col = next(
            (col for col in ['usage_start_time', 'date', 'timestamp'] 
            if col in df.columns),
            None
        )
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            df.sort_values(date_col, inplace=True)
        
        logger.info(
            f"Successfully loaded {len(df)} records from {path.name}\n"
            f"Columns: {list(df.columns)}\n"
            f"Date range: {df[date_col].min()} to {df[date_col].max() if date_col else 'N/A'}"
        )
        
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"Empty file: {path}")
        raise ValueError(f"File is empty: {path}")
        
    except Exception as e:
        logger.error(
            f"Failed to load {path.name}: {type(e).__name__}\n"
            f"Error: {str(e)}\n"
            f"File size: {path.stat().st_size if path.exists() else 0} bytes",
            exc_info=True
        )
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