# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# FIX: Install updated system dependencies for OpenCV
# We use libgl1 and libfribidi0 which are the modern replacements
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libfribidi0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app.py
COPY . .

# Run using gunicorn for production stability on Google Cloud
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
