# syntax=docker/dockerfile:1  
  
FROM python:3.11-slim  
  
ENV PYTHONUNBUFFERED=1  
  
# Set destination for COPY  
WORKDIR /app  
  
# Install system dependencies  
RUN apt-get update && apt-get install -y \  
    build-essential \  
    && rm -rf /var/lib/apt/lists/*  
  
# Copy Python requirements  
COPY container_src/requirements.txt ./  
  
# Install Python dependencies  
RUN pip install --no-cache-dir -r requirements.txt  
  
# Copy container source code  
COPY container_src/ ./  

# Expose port  
EXPOSE 8080  
  
# Run FastAPI with uvicorn  
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]