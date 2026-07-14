"""
Inference utility for predicting Harmful Algal Bloom (HAB) severity on unseen imagery.
Once train.py has finished training and saving a model, this script serves as the operational,
deployment pipeline to use in the real world. 

It accepts a single, brand-new satellite image,
extracts its statistical features, passes those features to the saved classifier, and 
translates the numerical prediction back into its original text-based severity class. 
In summary,
You hand it a brand-new, unseen satellite image, and it uses the saved model to predict how severe the algal bloom is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# joblib is used to load (deserialize) the trained model and label encoder objects from disk
import joblib
import numpy as np

# Reuses the exact same image reading and statistical extraction functions used during training.
from hab_pipeline.data_loader import _extract_features, _read_image_array


def predict_image(model_path: str, encoder_path: str, image_path: str) -> str:
    """
    Loads a trained classification pipeline, extracts features from a single target image,
    and returns a human-readable severity prediction ('Low', 'Moderate', or 'High').
    """
    # Step 1: Deserialize/load the trained classifier and label encoder from disk.
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    
    # Step 2: Load the satellite image as a standardized float32 RGB array.
    image_array = _read_image_array(Path(image_path))
    
    # Step 3: Compress raw pixel data into a 21-element statistical feature vector.
    raw_features = _extract_features(image_array)
    
    # Step 4: Reshape from 1D (21,) to 2D (1, 21) as scikit-learn expects 2D inputs for batch predictions.
    features = raw_features.reshape(1, -1)
    
    # Step 5: Feed the features to the model and retrieve the numerical class prediction index.
    prediction = model.predict(features)[0]
    
    # Step 6: Decode the predicted numerical class index back into its human-readable text label (e.g., 'High').
    decoded_label = encoder.inverse_transform([prediction])[0]
    return str(decoded_label)


# Entry point when executing the script directly from the command line
# e.g., python infer.py --image-path "/path/to/satellite_tile.tif"
if __name__ == "__main__":
    # Initialize the CLI argument parser with an overall description of the script's purpose
    parser = argparse.ArgumentParser(description="Predict HAB severity for one image")
    
    # --- ARGUMENT CONFIGURATION ---
    
   # 1. Path to the trained RandomForest file (defaults to training output path)
    parser.add_argument(
        "--model-path", 
        default="outputs/hab_model.joblib",
        help="Path to the trained RandomForest joblib model file"
    )
    
    # 2. Path to the LabelEncoder file (maps model's integer outputs back to text)
    parser.add_argument(
        "--encoder-path", 
        default="outputs/label_encoder.joblib",
        help="Path to the saved LabelEncoder joblib file"
    )
    
   # 3. Path to the target satellite image to classify (required for every run)
    parser.add_argument(
        "--image-path", 
        required=True,
        help="Path to the target satellite image to analyze"
    )
    
    # Parse the incoming arguments provided by the user in the command line
    args = parser.parse_args()
    
    # Trigger the prediction pipeline using the parsed arguments and print the final decoded severity level
    result = predict_image(args.model_path, args.encoder_path, args.image_path)
    print(f"\n🔮 Predicted HAB Severity: {result}")