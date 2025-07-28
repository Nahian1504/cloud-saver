# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file from documents folder
COPY documents/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into container
COPY . .

# Expose Streamlit default port
EXPOSE 7860

# Run Streamlit app from streamlit/app.py
CMD ["streamlit", "run", "streamlit/app.py", "--server.port=7860", "--server.address=0.0.0.0"]