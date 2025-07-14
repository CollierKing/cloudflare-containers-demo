import logging
import os
import numpy as np
import pandas as pd
import onnxruntime as ort

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_onnx_model(model_path):
    """Load the ONNX model for inference"""
    logger.info(f"Loading ONNX model from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create inference session
    session = ort.InferenceSession(model_path)
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Input name: {input_name}")
    logger.info(f"Output name: {output_name}")
    
    return session, input_name, output_name

def load_titanic_data(csv_path):
    """Load the processed Titanic data"""
    logger.info(f"Loading Titanic data from {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} records from CSV")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df

def select_random_record(df):
    """Select a random record from the dataframe"""
    random_record = df.sample(n=1).iloc[0]
    logger.info(f"Selected random record: {random_record.to_dict()}")
    return random_record

def prepare_input_data(record):
    """Prepare input data for ONNX model inference"""
    # Extract features in the correct order: [sex_encoded, age, pclass]
    features = [
        float(record['sex_encoded']),  # f0: sex_encoded (0=female, 1=male)
        float(record['age']),          # f1: age
        float(record['pclass'])        # f2: pclass (1, 2, or 3)
    ]
    
    # Convert to numpy array with correct shape and data type
    input_data = np.array([features], dtype=np.float32)
    
    logger.info(f"Input features: {features}")
    logger.info(f"Input shape: {input_data.shape}")
    
    return input_data

def run_inference(session, input_name, output_name, input_data):
    """Run inference using the ONNX model"""
    logger.info("Running inference...")
    
    # Run inference
    result = session.run([output_name], {input_name: input_data})
    prediction = result[0][0]  # Get the first (and only) prediction
    
    logger.info(f"Raw prediction: {prediction}")
    
    # Convert to survival prediction
    survival_prediction = int(prediction)
    survival_status = "survived" if survival_prediction == 1 else "died"
    
    logger.info(f"Prediction: {survival_status} (class: {survival_prediction})")
    
    return survival_prediction, survival_status

def interpret_record(record):
    """Interpret the record in human-readable format"""
    sex = "female" if record['sex_encoded'] == 0 else "male"
    age = record['age']
    pclass = int(record['pclass'])
    actual_survival = "survived" if record['survived'] == 1 else "died"
    
    passenger_info = f"{sex}, age {age:.1f}, {pclass}st/nd/rd class"
    
    return passenger_info, actual_survival