# Use the official Python 3.11 slim image as the base
FROM python:3.11-slim

# Set environment variables to prevent Python from writing pyc files and to buffer outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the Hugging Face cache directory to a known path
ENV HF_HOME=/app/hf_cache

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt into the container
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the summarization model during the build
RUN python -c "from transformers import pipeline; \
    pipeline('summarization', model='facebook/bart-large-cnn', framework='pt')"

# Copy the entire application code into the container
COPY . .

# Expose port 5000 for the Flask application
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Start the Gunicorn server with a single worker to reduce resource consumption
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "1", \
     "--timeout", "300", "--log-level", "debug", "--access-logfile", "-", "--error-logfile", "-"]