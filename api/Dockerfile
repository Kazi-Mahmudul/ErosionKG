# Use official Python runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if any needed for Neo4j driver or compilation)
RUN apt-get update && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variable for port (Cloud Run sets this to 8080 by default)
ENV PORT=8080

# Run the application
# Important: Bind to 0.0.0.0
CMD ["sh", "-c", "uvicorn index:app --host 0.0.0.0 --port ${PORT}"]
