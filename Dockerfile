# Use official Python base image
FROM python:3.11

# Install dependencies for FAISS
RUN apt-get update && apt-get install -y libopenblas-dev

# Set the working directory inside the container
WORKDIR /app

# Copy the necessary files
COPY requirements.txt .
COPY src/ ./src/
COPY data/ ./data/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for API
EXPOSE 8000

# Set default command to run FastAPI server
CMD ["python", "src/server.py"]
