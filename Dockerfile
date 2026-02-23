FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy all project files
COPY . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Start the server on port 7860
CMD ["uvicorn", "api.predict:app", "--host", "0.0.0.0", "--port", "7860"]
