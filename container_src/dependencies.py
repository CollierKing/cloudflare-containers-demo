import logging
import os
import onnxruntime as rt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def load_model(app):
    logger.info("Loading Titanic ONNX model...")

    # Load local Titanic ONNX model
    model_file = "model/titanic_xgboost_model.onnx"
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Titanic model file not found: {model_file}")
    
    # Load the ONNX model
    logger.info("Loading model from %s", model_file)
    session = rt.InferenceSession(model_file)
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Store in app state
    app.state.model = session
    app.state.input_name = input_name
    app.state.output_name = output_name
    
    logger.info(f"Titanic model loaded successfully")
    logger.info(f"Input name: {input_name}")
    logger.info(f"Output name: {output_name}")

