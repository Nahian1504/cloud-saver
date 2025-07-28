FROM python:3.11-slim

WORKDIR /app

# Install essential packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt from documents folder
COPY documents/requirements.txt ./requirements.txt

# Copy your Streamlit app folder (adjust if needed)
COPY streamlit/ ./streamlit/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Create writable directories for Streamlit config/cache/logs
RUN mkdir -p /app/.streamlit/cache /app/.streamlit/logs

# Set environment variables for Streamlit to avoid permission errors
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit
ENV STREAMLIT_CACHE_DIR=/app/.streamlit/cache
ENV STREAMLIT_LOGS_DIR=/app/.streamlit/logs

# Run the Streamlit app from streamlit/app.py
ENTRYPOINT ["streamlit", "run", "streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]