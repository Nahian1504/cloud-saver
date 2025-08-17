FROM python:3.11-slim

WORKDIR /app

# Install essential packages with retry logic for robustness
# This handles transient network issues by retrying the apt-get commands.
RUN set -ex; \
    for i in $(seq 1 5); do \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            build-essential \
            curl \
            software-properties-common \
            git && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* && \
        break || sleep 15; \
    done; \
    if [ $i -eq 5 ]; then \
        echo "Failed to install packages after 5 retries. Exiting."; \
        exit 1; \
    fi

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
