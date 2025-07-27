import os
import logging
from flask import Flask, request, jsonify
import torch
import numpy as np
import sys
import socket
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.sac_agent_model import SACAgent

# ========== Configuration ==========
STATE_DIM = 4
ACTION_DIM = 1
MODEL_PATH = os.path.join('models', 'sac_agent.pth')
PORT = 5001  # Using fixed port for easier debugging
HOST = '127.0.0.1'  # Using localhost first for security

# ========== Logging Setup ==========
def configure_logging():
    """Configure comprehensive logging"""
    os.makedirs('logs', exist_ok=True)
    log_filename = f"logs/api_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.DEBUG,  # More verbose during debugging
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = configure_logging()

# ========== Flask Application ==========
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ========== SAC Agent Initialization ==========
def initialize_agent():
    """Initialize and verify the SAC agent"""
    global agent
    
    logger.info("===== Initializing SAC Agent =====")
    logger.info(f"Model path: {os.path.abspath(MODEL_PATH)}")
    
    try:
        # Verify model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        # Initialize agent
        agent = SACAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
        logger.info(f"Agent initialized with state_dim={STATE_DIM}, action_dim={ACTION_DIM}")
        
        # Load weights
        agent.load(MODEL_PATH)
        logger.info("Model weights loaded successfully")
        
        # Test inference
        test_state = np.random.randn(STATE_DIM)
        with torch.no_grad():
            action = agent.select_action(test_state, evaluate=True)
            logger.info(f"Test prediction successful - Input: {test_state}, Output: {action}")
            
        return True
        
    except Exception as e:
        logger.critical(f"Agent initialization failed: {str(e)}", exc_info=True)
        return False

# ========== API Endpoints ==========
@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    try:
        status = {
            'api_status': 'running',
            'model_loaded': agent is not None,
            'model_exists': os.path.exists(MODEL_PATH),
            'state_dim': STATE_DIM,
            'action_dim': ACTION_DIM,
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            'python_version': sys.version,
            'torch_version': torch.__version__
        }
        logger.debug("Health check requested")
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint with thorough validation"""
    try:
        if agent is None:
            raise RuntimeError("Agent not initialized")
            
        # Validate input
        if not request.is_json:
            raise ValueError("Request must be JSON")
            
        data = request.get_json()
        state = data.get('state')
        
        if state is None:
            raise ValueError("'state' parameter is required")
            
        # Convert and validate state
        try:
            state_array = np.array(state, dtype=np.float32).reshape(-1)
            if len(state_array) != STATE_DIM:
                raise ValueError
        except:
            raise ValueError(f"State must be a 1D array of length {STATE_DIM}")
        
        # Get prediction
        with torch.no_grad():
            action = agent.select_action(state_array, evaluate=True)
            action = action.tolist()
        
        logger.info(f"Prediction successful - State: {state_array.tolist()}")
        return jsonify({
            'status': 'success',
            'input_state': state_array.tolist(),
            'predicted_action': action,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'expected_format': {
                'state': f"Array of length {STATE_DIM}",
                'example': {'state': [0.1, 0.2, 0.3, 0.4]}
            }
        }), 400

# ========== Server Startup ==========
def check_port_availability(port, host='127.0.0.1'):
    """Check if port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0

if __name__ == '__main__':
    try:
        logger.info("\n===== Starting Cloud Saver API =====")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Torch version: {torch.__version__}")
        logger.info(f"Running on: {HOST}:{PORT}")
        
        # Verify port availability
        if not check_port_availability(PORT, HOST):
            raise RuntimeError(f"Port {PORT} is already in use")
        
        # Initialize agent
        if not initialize_agent():
            raise RuntimeError("Agent initialization failed")
        
        # Print startup banner
        print("\n" + "="*50)
        print(f"Cloud Saver API - Ready")
        print(f"• Local URL: http://{HOST}:{PORT}/health")
        print(f"• Model: {os.path.basename(MODEL_PATH)}")
        print(f"• State dim: {STATE_DIM}, Action dim: {ACTION_DIM}")
        print("="*50 + "\n")
        
        # Start server
        app.run(
            host=HOST,
            port=PORT,
            debug=False,  # Disable debug mode for production
            use_reloader=False
        )
        
    except Exception as e:
        logger.critical(f"Fatal startup error: {str(e)}", exc_info=True)
        print(f"\n!!! SERVER FAILED TO START !!!")
        print(f"Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print(f"1. Check if port {PORT} is free (netstat -ano | findstr :{PORT})")
        print("2. Verify the model exists at:", os.path.abspath(MODEL_PATH))
        print("3. Check logs/api_*.log for details")
        input("\nPress Enter to exit...")
        sys.exit(1)