import numpy as np
import pandas as pd
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx

def load_titanic_data():
    """Load the Titanic dataset from OpenML"""
    print("Loading Titanic dataset...")
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    
    # Select only the required columns
    selected_columns = ['sex', 'age', 'pclass']
    X_selected = X[selected_columns].copy()
    
    print(f"Dataset shape: {X_selected.shape}")
    print(f"Columns: {X_selected.columns.tolist()}")
    print(f"Missing values per column:")
    print(X_selected.isnull().sum())
    
    return X_selected, y

def preprocess_data(X, y):
    """Preprocess the data for XGBoost"""
    print("\nPreprocessing data...")
    
    # Handle missing values in age using median imputation
    imputer = SimpleImputer(strategy='median')
    X['age'] = imputer.fit_transform(X[['age']])
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    X['sex_encoded'] = label_encoder.fit_transform(X['sex'])
    
    # Create final feature matrix
    X_processed = X[['sex_encoded', 'age', 'pclass']].copy()
    
    # Convert target to binary (0 for died, 1 for survived)
    y_processed = y.astype(int)
    
    print(f"Processed data shape: {X_processed.shape}")
    print(f"Target distribution: {y_processed.value_counts().to_dict()}")
    
    return X_processed, y_processed

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier"""
    print("\nTraining XGBoost classifier...")
    
    # Initialize XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train the model
    xgb_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = xgb_classifier.predict(X_test)
    y_pred_proba = xgb_classifier.predict_proba(X_test)
    
    return xgb_classifier, y_pred, y_pred_proba

def evaluate_model(y_test, y_pred, feature_names, model):
    """Evaluate the trained model"""
    print("\nModel Evaluation:")
    print("="*50)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    print("\nFeature Importance:")
    feature_importance = model.feature_importances_
    for i, feature in enumerate(feature_names):
        print(f"{feature}: {feature_importance[i]:.4f}")

def save_model_as_onnx(model, feature_names, model_dir="model"):
    """Save the trained XGBoost model as ONNX format"""
    print(f"\nSaving model as ONNX...")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Fix feature naming for ONNX - rename features to f0, f1, f2 format
    # XGBoost expects feature names like f0, f1, f2 for ONNX conversion
    onnx_feature_names = [f'f{i}' for i in range(len(feature_names))]
    
    # Set feature names in the XGBoost booster
    model.get_booster().feature_names = onnx_feature_names
    
    # Define the input type for ONNX conversion
    initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
    
    try:
        # Convert the XGBoost model to ONNX format
        onnx_model = convert_xgboost(model, initial_types=initial_type)
        
        # Save the ONNX model
        onnx_file_path = os.path.join(model_dir, "titanic_xgboost_model.onnx")
        onnx.save_model(onnx_model, onnx_file_path)
        
        print(f"Model successfully saved as ONNX at: {onnx_file_path}")
        
        # Save feature names mapping for later use
        feature_names_file = os.path.join(model_dir, "feature_names.txt")
        with open(feature_names_file, 'w') as f:
            f.write("Feature Name Mapping (for ONNX model):\n")
            f.write("=====================================\n")
            for i, feature in enumerate(feature_names):
                f.write(f"f{i}: {feature}\n")
        
        print(f"Feature names mapping saved at: {feature_names_file}")
        
        # Save model metadata
        metadata_file = os.path.join(model_dir, "model_metadata.txt")
        with open(metadata_file, 'w') as f:
            f.write("XGBoost Titanic Classifier\n")
            f.write("=========================\n")
            f.write(f"Original Features: {', '.join(feature_names)}\n")
            f.write(f"ONNX Features: {', '.join(onnx_feature_names)}\n")
            f.write(f"Model type: XGBoost Classifier\n")
            f.write(f"Input shape: [batch_size, {len(feature_names)}]\n")
            f.write(f"Output: Binary classification (0=died, 1=survived)\n")
            f.write(f"Feature encoding: f0=sex_encoded (0=female, 1=male), f1=age, f2=pclass\n")
        
        print(f"Model metadata saved at: {metadata_file}")
        
        return onnx_file_path
        
    except Exception as e:
        print(f"Error saving model as ONNX: {str(e)}")
        return None


def main():
    """Main function to run the XGBoost training pipeline"""
    print("XGBoost Titanic Classifier")
    print("="*50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    X, y = load_titanic_data()
    
    # Preprocess data
    X_processed, y_processed = preprocess_data(X, y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_processed
    )
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    model, y_pred, y_pred_proba = train_xgboost_model(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    feature_names = ['sex_encoded', 'age', 'pclass']
    evaluate_model(y_test, y_pred, feature_names, model)
    
    # Save model as ONNX
    onnx_path = save_model_as_onnx(model, feature_names)
    
    # Export processed data to CSV
    os.makedirs("model", exist_ok=True)
    processed_data = X_processed.copy()
    processed_data['survived'] = y_processed
    processed_data.to_csv("model/titanic_processed_data.csv", index=False)
    print(f"Processed data exported to: model/titanic_processed_data.csv")
    
    # Display some predictions
    print("\nSample Predictions:")
    print("="*50)
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    for i in sample_indices:
        sex = "male" if X_test.iloc[i]['sex_encoded'] == 1 else "female"
        age = X_test.iloc[i]['age']
        pclass = X_test.iloc[i]['pclass']
        prediction = "survived" if y_pred[i] == 1 else "died"
        confidence = max(y_pred_proba[i]) * 100
        actual = "survived" if y_test.iloc[i] == 1 else "died"
        
        print(f"Sex: {sex}, Age: {age:.1f}, Class: {pclass} -> Predicted: {prediction} ({confidence:.1f}% confidence), Actual: {actual}")

if __name__ == "__main__":
    main()
