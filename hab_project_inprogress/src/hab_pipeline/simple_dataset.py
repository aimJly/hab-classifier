"""
This is a stripped-down, beginner-friendly version of the data loader. It looks specifically for .tif image files, 
extracts basic statistics, and assigns a simple 1 or 0 label based on whether the word "bloom" is in the file name.
"""

"""
This is a stripped-down, beginner-friendly version of the data loader. It looks specifically for .tif image files, 
extracts basic statistics, and assigns a simple 1 or 0 label based on whether the word "bloom" is in the file name.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image


def discover_hab_images(root: str | os.PathLike[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Scans a directory for satellite imagery (.tif files), extracts statistical color features 
    for each image, and assigns a binary label based on the file path name.

    Parameters:
    -----------
    root : str or os.PathLike
        The root directory path where the script should search for image files.

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, pd.DataFrame]
        - A 2D NumPy array of extracted features (shape: [num_images, 21])
        - A 1D NumPy array of binary labels (shape: [num_images])
        - A Pandas DataFrame mapping the successful image paths to their labels
    """
    # Convert the input path string into a robust Path object for easier cross-platform traversal
    root_path = Path(root)
    
    # Recursively find all '.tif' files in the root folder and any subfolders, then sort them alphabetically
    image_files = sorted(root_path.rglob("*.tif"))
    
    # If no images are found, stop execution immediately and tell the user where it looked
    if not image_files:
        raise FileNotFoundError(f"No TIFF files found under {root_path}")

    samples = []
    labels = []
    
    # Limit processing to the first 200 images to prevent memory overflow and keep execution fast
    for image_file in image_files[:200]:
        try:
            # Open the image file and explicitly convert it to the 3-channel RGB standard format
            image = Image.open(image_file).convert("RGB")
        except Exception:
            # If an image is corrupted or cannot be read, silently skip it and move to the next file
            continue
            
        # Convert the pixel values to a NumPy matrix of decimals (float32) for mathematical modeling
        array = np.array(image, dtype=np.float32)
        
        # Flatten the 2D pixel grid into a 2D list of pixels while preserving the 3 color channels (Red, Green, Blue)
        # Shape changes from (Height, Width, 3) -> (Total Pixels, 3)
        flat = array.reshape(-1, 3)
        
        # Feature Extraction Pipeline:
        # Instead of feeding thousands of raw pixels into a model, we compress each image down to 21 statistical summary numbers:
        # - 3 Mean values (average brightness of Red, Green, and Blue)
        # - 3 Std values (how much the color varies across the image for Red, Green, and Blue)
        # - 15 Percentile values (captures structural lighting trends: 5th, 25th, 50th, 75th, 95th percentiles across all 3 channels)
        features = np.concatenate([
            flat.mean(axis=0),                                          # Average per channel (3 values)
            flat.std(axis=0),                                           # Standard deviation per channel (3 values)
            np.percentile(flat, [5, 25, 50, 75, 95], axis=0).reshape(-1), # Spreading metrics per channel (15 values)
        ]).astype(np.float32)
        
        # Save the computed 21-element numerical blueprint of this image
        samples.append(features)
        
        # Rule-based Labeling Heuristic:
        # Assign a 1 (Harmful Algal Bloom Present) if the file path or name includes the word "bloom".
        # Otherwise, assign a 0 (Normal Water / No Bloom).
        labels.append(1 if "bloom" in str(image_file).lower() else 0)

    # Consolidate the valid image file paths and labels into a clean tabular layout
    df = pd.DataFrame({
        "image_path": [str(p) for p in image_files[: len(samples)]], 
        "label": labels
    })
    
    # Stack the individual 1D feature arrays vertically into a unified 2D dataset matrix (Rows = Samples, Columns = Features)
    # Convert the labels into a standardized 1D array, and return everything alongside the tracker DataFrame
    return np.vstack(samples), np.asarray(labels), df