# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# FIX: Install updated system dependencies for OpenCV
# We use libgl1 and libglib2.0-0 to prevent the 'package not found' error
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
# IMPORTANT: Ensure requirements.txt is in your folder before building
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application (app.py)
COPY . .

# Start the application using Gunicorn
# This handles the $PORT requirement for Google Cloud Run
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
