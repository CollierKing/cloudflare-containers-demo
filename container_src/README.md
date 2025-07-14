# Titanic Survival Prediction API

A FastAPI application that predicts Titanic passenger survival using an XGBoost model exported to ONNX format.

## Features

- **XGBoost Model**: Trained on the Titanic dataset with 85.5% accuracy
- **ONNX Runtime**: Fast inference using ONNX format
- **FastAPI**: Modern web API with automatic documentation
- **RESTful Endpoints**: Easy-to-use prediction API

## Project Structure

```
container_src/
├── model/
│   ├── titanic_xgboost_model.onnx    # Trained ONNX model
│   ├── titanic_processed_data.csv    # Processed training data
│   ├── feature_names.txt             # Feature mapping
│   ├── model_metadata.txt            # Model information
│   └── train_model.py                # Model training script
├── main.py                           # FastAPI application
├── dependencies.py                   # Model loading utilities
├── utils.py                          # Utility functions
├── proc.py                           # Batch inference script
├── config.py                         # Configuration settings
├── pyproject.toml                    # UV project file
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Setup Instructions

### 1. Install Dependencies

#### Using UV (Recommended)
```bash
# Install UV if you haven't already
pip install uv

# Install project dependencies
uv sync
```

#### Using pip
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)

The model is already trained, but you can retrain it:

```bash
# Train the XGBoost model and export to ONNX
uv run python model/train_model.py
```

This will:
- Download the Titanic dataset from OpenML
- Train an XGBoost classifier on features: sex, age, pclass
- Export the model to ONNX format
- Save processed data and metadata

### 3. Run the API Server

```bash
# Start the FastAPI server
uv run python main.py
```

The API will be available at `http://localhost:8080`

### 4. Test the API

```bash
# Test inference endpoint
curl -X POST "http://localhost:8080/api/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "sex": "female",
    "age": 25,
    "pclass": 1
  }'
```

## API Endpoints

### Root Endpoint
- **GET** `/` - API information and available endpoints

### Health Check
- **GET** `/api/health` - Service health status

### Inference
- **POST** `/api/inference` - Predict passenger survival

#### Request Format
```json
{
  "sex": "male" or "female",
  "age": float,
  "pclass": 1, 2, or 3
}
```

#### Response Format
```json
{
  "prediction": 0 or 1,
  "prediction_status": "died" or "survived",
  "passenger_info": {
    "sex": "male",
    "age": 25.0,
    "pclass": 3
  }
}
```

## Model Information

- **Algorithm**: XGBoost Classifier
- **Features**: 
  - `sex` (encoded: 0=female, 1=male)
  - `age` (numeric, missing values imputed with median)
  - `pclass` (passenger class: 1, 2, or 3)
- **Target**: Binary classification (0=died, 1=survived)
- **Accuracy**: 85.5% on test set
- **Format**: ONNX for fast inference

### Feature Importance
1. **Sex**: 87.95% - Most important factor
2. **Passenger Class**: 9.99% - Moderately important
3. **Age**: 2.06% - Least important

## Development

### Export Dependencies
```bash
# Export UV dependencies to requirements.txt (without hashes for cleaner output)
uv export --format requirements-txt --no-hashes --output-file requirements.txt
```

### Run Batch Inference
```bash
# Test model with random passengers
uv run python proc.py
```

### Interactive Testing
```bash
# Use the test script
uv run python ../test_scripts/test-fastapi.py
```

## Example Usage

```python
import requests

# Test different passenger scenarios
test_cases = [
    {"sex": "female", "age": 25, "pclass": 1},  # Young female, 1st class
    {"sex": "male", "age": 50, "pclass": 1},    # Older male, 1st class  
    {"sex": "female", "age": 35, "pclass": 3},  # Female, 3rd class
    {"sex": "male", "age": 8, "pclass": 2}      # Child male, 2nd class
]

for passenger in test_cases:
    response = requests.post("http://localhost:8080/api/inference", json=passenger)
    result = response.json()
    print(f"{passenger} -> {result['prediction_status']}")
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8080/docs`
- **ReDoc**: `http://localhost:8080/redoc`

## Requirements

- Python 3.11+
- FastAPI
- ONNX Runtime
- XGBoost
- Pandas
- NumPy
- Scikit-learn

## License

This project is licensed under the MIT License - see the LICENSE file for details.
