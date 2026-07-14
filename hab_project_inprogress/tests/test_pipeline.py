"""
This is the safety inspector. It contains automated tests to ensure that the data loader and the training script are functioning correctly. 
It creates fake, temporary images to verify that the pipeline can process them without crashing.
"""
from __future__ import annotations

import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from hab_pipeline.data_loader import load_dataset
from hab_pipeline.train import train_model


def test_load_dataset_and_train(tmp_path):
    """
    Tests the complete data loading and model training pipeline.
    It creates a temporary mock dataset with fake images and a labels.csv file,
    then runs the data loader and training loop to ensure everything works end-to-end.
    """
    # 'tmp_path' is a special pytest fixture that gives us a temporary, clean directory 
    # for this test run. It gets deleted automatically later.
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Generate 4 tiny (16x16 pixel) solid-color images to act as fake satellite data
    for name, color in [("a.png", (255, 0, 0)), ("b.png", (0, 255, 0)), ("c.png", (0, 0, 255)), ("d.png", (255, 255, 0))]:
        image = Image.new("RGB", (16, 16), color)
        image.save(image_dir / name)

    # Create a mock CSV file linking each image to a fake severity level (1 through 4)
    labels_path = tmp_path / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "severity"])
        writer.writerow(["images/a.png", 1])
        writer.writerow(["images/b.png", 2])
        writer.writerow(["images/c.png", 3])
        writer.writerow(["images/d.png", 4])

    # Test the data loader: Ensure it correctly finds and loads the 4 images and labels
    X, y, metadata = load_dataset(data_dir=str(tmp_path))
    
    # Assertions are checks that will fail the test if they aren't True
    assert X.shape[0] == 4  # X contains our image features (4 rows of data)
    assert y.shape[0] == 4  # y contains our labels (4 labels)
    assert "image_path" in metadata.columns

    # Test the training loop: Pass the mock data through the Random Forest model
    output_dir = tmp_path / "outputs"
    metrics = train_model(data_dir=str(tmp_path), output_dir=str(output_dir), test_size=0.5, random_state=7)
    
    # Verify the model actually trained and saved the expected output files
    assert metrics["accuracy"] >= 0.0
    assert (output_dir / "hab_model.joblib").exists()


def test_load_dataset_from_folder_metadata(tmp_path):
    """
    Tests an alternative data loading method where metadata (like labels) 
    is stored in a JSON file inside individual sample folders, rather than a single master CSV.
    """
    # Create a mock folder structure for a single dataset sample
    sample_dir = tmp_path / "data" / "sample_001"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a dummy satellite image for this specific sample
    image = Image.new("RGB", (16, 16), (8, 16, 24))
    image.save(sample_dir / "visual_raw.png")

    # Create a mock JSON payload containing the "High" severity label
    # The original image dataset doesn't have labels for each image
    metadata_payload = {
        "uid": "sample_001",
        "indicative_class": "High",
        "severity": "High",
    }
    
    # Write the JSON payload to a metadata.json file sitting right next to the image
    (sample_dir / "metadata.json").write_text(__import__("json").dumps(metadata_payload), encoding="utf-8")

    # Test the data loader: Ensure it can read the folder structure and parse the JSON for labels
    X, y, metadata = load_dataset(data_dir=str(tmp_path))
    
    # Verify the loader found exactly 1 sample with the correct label
    assert X.shape[0] == 1
    assert y.shape[0] == 1
    assert metadata.iloc[0]["label"] == "High"
    assert metadata.iloc[0]["image_path"].endswith("visual_raw.png")