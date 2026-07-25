import os
import joblib
import pandas as pd
import numpy as np

# Resolve path to artifacts directory (relative to this file)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(os.path.dirname(BACKEND_DIR), 'artifacts')

MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'model.pkl')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, 'encoder.pkl')

class PredictionEngine:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.encoder_meta = None
        self.is_loaded = False

    def load_artifacts(self):
        """Loads model, preprocessor, and encoder metadata from artifacts."""
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(
                f"Required artifact files not found in {ARTIFACTS_DIR}. "
                "Ensure that the training pipeline has run and generated model.pkl, scaler.pkl, and encoder.pkl."
            )
        
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(SCALER_PATH)
        self.encoder_meta = joblib.load(ENCODER_PATH)
        self.is_loaded = True

    def predict(self, input_data: dict) -> float:
        """
        Preprocesses inputs and predicts the annual medical cost.
        
        Args:
            input_data (dict): Feature key-value pairs representing demographic, health, and policy details.
            
        Returns:
            float: Predicted medical insurance cost (annual charges).
        """
        if not self.is_loaded:
            self.load_artifacts()
            
        # Convert dictionary input to Pandas DataFrame
        df = pd.DataFrame([input_data])
        
        # Apply ordinal mappings
        ordinal_mappings = self.encoder_meta['ordinal_mappings']
        for col, mapping in ordinal_mappings.items():
            if col in df.columns:
                val = df.at[0, col]
                if pd.isna(val) or val is None:
                    df[col] = mapping.get('None', 0)
                else:
                    df[col] = df[col].map(mapping)
                    
        # Apply preprocessor (scaler and one-hot encoding for nominal categories)
        processed_features = self.preprocessor.transform(df)
        
        # Make prediction
        prediction = self.model.predict(processed_features)
        
        # Return scalar float
        return float(prediction[0])

# Instantiate a single engine instance for import
engine = PredictionEngine()
