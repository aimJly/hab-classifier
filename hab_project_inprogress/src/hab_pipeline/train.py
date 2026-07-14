"""
It takes the prepared data from the data_loader.py and splits it: one pile for "studying" 
(training data) and one pile for "testing" (testing data). It uses an algorithm called a Random Forest to 
learn the patterns of algal blooms. Finally, it tests itself, saves the finished brain (the model), and draws a 
chart (a confusion matrix) to show how accurate it was.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

# joblib is used to save and load trained machine learning models to/from files
import joblib

# matplotlib and seaborn are libraries used for creating graphs and charts
import matplotlib
# "Agg" tells matplotlib to generate plots in the background without opening a popup window
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# scikit-learn (sklearn) is the primary toolkit used here for machine learning algorithms and evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from hab_pipeline.data_loader import load_dataset


def train_model(
    dataset_name: str = "kostaspic/amfitrite-inland-waters-hab-sentinel2",
    data_dir: Optional[str] = None,
    output_dir: str = "outputs",
    test_size: float = 0.2,
    random_state: int = 42,
    split: str = "train",
    limit: Optional[int] = None,
) -> dict:
    """
    The main machine learning pipeline. It:
    1. Loads the satellite imagery features and labels.
    2. Converts text labels ("High", "Low") into numbers.
    3. Splits data into a "studying" set and a "testing" set.
    4. Trains a Random Forest AI model.
    5. Evaluates how well the model performed and saves the results.
    """
    
    # Step 1: Load the dataset using the logic defined in data_loader.py
    # X = features (the math stats of the images), y = targets/labels (bloom severity)
    X, y, metadata = load_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split=split,
        limit=limit,
    )

    # Step 2: Convert words into numbers
    # Machine learning algorithms can't read words like "High", "Moderate", or "Low".
    # LabelEncoder translates them into 0, 1, and 2 so the computer can compute them.
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y.astype(str))

    # Step 3: Split the data into Training and Testing sets
    # We hide 20% (test_size=0.2) of the data so we can test the model on unseen images later.
    # 'stratify' ensures that if 10% of our original data is "High severity", both our 
    # training and testing piles will also contain exactly 10% "High severity" samples.
    # 'random_state' is a seed number that ensures the random shuffle happens the exact same way every time.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None,
    )

    # Step 4: Define the Machine Learning Brain (Random Forest)
    # A Random Forest works by creating many individual "Decision Trees" (in this case, 200) 
    # that all look at the data and vote on whether they think a bloom is High, Moderate, or Low.
    # 'class_weight="balanced_subsample"' forces the model to pay extra attention to categories 
    # that are rare in the dataset so it doesn't get lazy and just guess the most common category.
    # 'n_jobs=-1' tells Python to use all available processing cores on your computer to speed things up.
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    
    # The .fit() function is the actual "learning" or training phase. 
    # The model studies the X_train features and their corresponding y_train answers.
    model.fit(X_train, y_train)

    # Step 5: Evaluation
    # We ask the trained model to guess the answers for the X_test data it has never seen before.
    preds = model.predict(X_test)
    
    # Calculate performance scores:
    # - accuracy: overall percentage of correct guesses.
    # - balanced_accuracy: adjusts the score if one category is much more common than others.
    # - macro_f1: evaluates how well the model handles all individual classes equally.
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, preds)), 4),
        "macro_f1": round(float(f1_score(y_test, preds, average="macro")), 4),
        "classes": encoder.classes_.tolist(), # Saves the original names of the classes (e.g., ["High", "Low"])
    }

    # Step 6: Save the results
    # Create the output folder if it doesn't exist yet
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the trained model brain and the label encoder to files so infer.py can use them later
    joblib.dump(model, output_path / "hab_model.joblib")
    joblib.dump(encoder, output_path / "label_encoder.joblib")
    joblib.dump({"feature_count": int(X.shape[1])}, output_path / "feature_metadata.joblib")

    # Save the math scores as a clean text file (.json)
    with open(output_path / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    # Step 7: Generate a Confusion Matrix chart
    # A confusion matrix is a grid that charts "What the real category was" vs "What the model guessed".
    # It makes it incredibly easy to see if the model is constantly misclassifying "Moderate" as "Low", etc.
    cm = confusion_matrix(y_test, preds, labels=np.arange(len(encoder.classes_)))
    plt.figure(figsize=(6, 5))
    
    # sns.heatmap draws a colored grid where darker squares represent higher numbers of predictions
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    # Save the grid chart as an image file
    plt.savefig(output_path / "confusion_matrix.png")
    plt.close()

    # Save a small CSV file summarizing the data used for training
    metadata.to_csv(output_path / "dataset_preview.csv", index=False)
    return metrics


def parse_args() -> argparse.Namespace:
    """
    Configures command-line arguments. 
    This allows a user to run this script from their terminal and pass custom configurations,
    for example: python train.py --test-size 0.3 --output-dir "my_run_outputs"
    """
    parser = argparse.ArgumentParser(description="Train an HAB severity classifier")
    parser.add_argument("--dataset-name", default="kostaspic_amfitrite")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


# This line checks if the script is being run directly by the user (e.g., double-clicked or run via terminal)
# If it is, it triggers the argument parser and starts the training pipeline automatically.
if __name__ == "__main__":
    print("\n🚀 [START] train.py has successfully started execution!")
    args = parse_args()
    
    print("⚙️ Arguments parsed successfully. Launching training pipeline...")
    metrics = train_model(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        split=args.split,
        limit=args.limit,
    )
    
    print("\n🎉 [SUCCESS] Training complete! Here are your metrics:")    
    # Print out the final scores to the screen so the user can see how well it did
    print(json.dumps(metrics, indent=2))