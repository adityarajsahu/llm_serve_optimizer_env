FROM ghcr.io/meta-pytorch/openenv-base:latest

# Set working directory
WORKDIR /app

# Copy all environment files
COPY requirements.txt /tmp/requirements.txt
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Expose the default OpenEnv port
EXPOSE 8000

# Start the FastAPI server via uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]