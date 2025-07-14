from utils import load_titanic_data, load_onnx_model, select_random_record, interpret_record, prepare_input_data, run_inference
import logging

logger = logging.getLogger(__name__)


def main():
    """Main function to run Titanic survival prediction"""
    logger.info("Starting Titanic Survival Prediction")
    logger.info("=" * 50)
    
    # Define file paths
    model_path = "model/titanic_xgboost_model.onnx"
    csv_path = "model/titanic_processed_data.csv"
    
    try:
        # Load ONNX model
        session, input_name, output_name = load_onnx_model(model_path)
        
        # Load Titanic data
        df = load_titanic_data(csv_path)
        
        # Select random record
        random_record = select_random_record(df)
        
        # Interpret the record
        passenger_info, actual_survival = interpret_record(random_record)
        
        # Prepare input data
        input_data = prepare_input_data(random_record)
        
        # Run inference
        predicted_class, predicted_status = run_inference(
            session, input_name, output_name, input_data
        )
        
        # Display results
        logger.info("\nPrediction Results:")
        logger.info("=" * 50)
        logger.info(f"Passenger: {passenger_info}")
        logger.info(f"Predicted: {predicted_status}")
        logger.info(f"Actual: {actual_survival}")
        
        # Check if prediction is correct
        is_correct = predicted_class == random_record['survived']
        accuracy_status = "✓ Correct" if is_correct else "✗ Incorrect"
        logger.info(f"Prediction accuracy: {accuracy_status}")
        
        logger.info("\nInference completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()


