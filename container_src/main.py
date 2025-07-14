import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi import Request
from dependencies import load_model

# Get environment variables
API_TITLE = os.getenv("API_TITLE", "FastAPI Container")
API_VERSION = os.getenv("API_VERSION", "1.0.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "prod")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# MARK: - LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize resources
    logger.info("Initializing model dependencies on application startup...")
    await load_model(app)
    logger.info("Graph initialization completed successfully")

    yield

    # Shutdown: cleanup resources if needed
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Titanic Survival Prediction API using XGBoost and ONNX Runtime",
    lifespan=lifespan,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)


# MARK: - SCHEMAS
class TitanicInferenceRequest(BaseModel):
    sex: str  # "male" or "female"
    age: float
    pclass: int  # 1, 2, or 3


# MARK: - ROUTES
@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Titanic Survival Prediction API!",
        "title": API_TITLE,
        "version": API_VERSION,
        "environment": ENVIRONMENT,
        "container_id": os.getenv("CLOUDFLARE_DEPLOYMENT_ID", "unknown"),
        "model_info": {
            "model_type": "XGBoost Titanic Survival Classifier",
            "features": ["sex", "age", "pclass"],
            "output": "Binary classification (0=died, 1=survived)"
        },
        "endpoints": {
            "GET /": "This endpoint",
            "GET /api/health": "Health check",
            "POST /api/inference": "Titanic survival prediction",
            "GET /api/container-info": "Container information"
        }
    }


@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": API_TITLE,
        "version": API_VERSION,
        "container_id": os.getenv("CLOUDFLARE_DEPLOYMENT_ID", "unknown")
    }


@app.get("/api/container-info")
def container_info():
    """Get container information"""
    return {
        "container_id": os.getenv("CLOUDFLARE_DEPLOYMENT_ID", "unknown"),
        "environment": ENVIRONMENT,
        "api_title": API_TITLE,
        "api_version": API_VERSION,
    }


# MARK: - INFERENCE
@app.post("/api/inference")
def inference(infer_req: TitanicInferenceRequest, request: Request):
    logger.info(f"Scoring: {infer_req.model_dump()}")

    # Get Model
    session = request.app.state.model
    input_name = request.app.state.input_name
    output_name = request.app.state.output_name

    # Convert sex string to encoded value
    sex_encoded = 0.0 if infer_req.sex.lower() == "female" else 1.0
    
    # Create a record-like object for the utility functions
    record = {
        'sex_encoded': sex_encoded,
        'age': float(infer_req.age),
        'pclass': float(infer_req.pclass)
    }
    
    # Use utility functions to prepare input data and run inference
    from utils import prepare_input_data, run_inference
    
    # Prepare input data
    input_data = prepare_input_data(record)
    
    # Run inference
    prediction_class, prediction_status = run_inference(
        session, input_name, output_name, input_data
    )
    
    logger.info(f"Titanic Inference result: {prediction_status}")
    
    return {
        "prediction": prediction_class,
        "prediction_status": prediction_status,
        "passenger_info": {
            "sex": infer_req.sex,
            "age": infer_req.age,
            "pclass": infer_req.pclass
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

