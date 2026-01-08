# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and YOLO
# libgl1-mesa-glx and libglib2.0-0 are essential for cv2 to run in Linux
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (app.py)
COPY . .

# Create the uploads folder (though Cloud Run uses temporary storage)
RUN mkdir -p uploads

# Start the application using Gunicorn for better production performance
# It listens on the port provided by Google Cloud ($PORT)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
