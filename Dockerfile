FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install PyTorch CPU first to save massive amounts of space (avoids downloading CUDA toolkits)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory for HuggingFace embeddings
ENV TRANSFORMERS_CACHE=/app/.cache
RUN mkdir -p /app/.cache && chmod 777 /app/.cache

# Copy application files
COPY . .

# Ensure data and vectorstore directories exist and are writable
RUN mkdir -p /app/backend/data /app/backend/vectorstore && \
    chmod -R 777 /app/backend/data /app/backend/vectorstore

# Expose port (HuggingFace Spaces uses 7860 by default)
EXPOSE 7860

# Run the FastAPI server using Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
