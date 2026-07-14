"""
Data loading and feature extraction utilities for HAB (Harmful Algal Bloom) satellite imagery.
This file acts as a versatile bridge between raw image files on disk and ready-to-train NumPy arrays.
It supports three primary local file structures (sample-subfolder catalogs, flat table mappings, 
or unstructured directory crawls) as well as remote streaming from the Hugging Face Hub.

It searches your folders (or the Hugging Face internet database) 
for satellite images. It then opens those images, extracts mathematical summaries (like the mean and standard deviation 
of the pixel colors), and matches them to their correct labels (High, Moderate, Low).
"""

from __future__ import annotations

import json
import os
import warnings
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:
    import tifffile
except Exception:  # pragma: no cover - optional dependency
    # tifffile is an optional dependency used specifically for reading multi-band TIFF satellite images
    tifffile = None


def _read_image_array(path: Path) -> np.ndarray:
    """
    Reads an image from disk and returns it as a standardized RGB float32 array.

    TIFF imagery requires specialized reading due to potentially higher bit depths or custom
    band counts. This function automatically converts multi-spectral (more than 3 channels) 
    or grayscale (1-channel or 2D) matrices into a standard 3-channel RGB matrix format.
    """
    suffix = path.suffix.lower()
    
    # Check if the image is a TIFF format. These require 'tifffile' rather than PIL.
    if suffix in {".tif", ".tiff"}:
        if tifffile is None:
            raise ImportError("tifffile is required to read TIFF images")
        
        # Read raw multi-spectral arrays without clamping dynamic range
        arr = tifffile.imread(path)
        
        # Scenario A: Grayscale/2D image. Stacks it three times to construct pseudo-RGB bands.
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
            
        # Scenario B: 3D array with a single channel in the last axis. Repeats it to build 3 channels.
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
            
        # Scenario C: Multi-spectral array (e.g., 4 or more bands). Discards extra bands (like near-infrared)
        # to focus on the standard Red, Green, and Blue bands (the first three channels).
        elif arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]
            
        return arr.astype(np.float32)

    # Fallback: Open standard consumer formats (PNG, JPEG, BMP) and force convert them to 8-bit RGB.
    image = Image.open(path).convert("RGB")
    return np.array(image, dtype=np.float32)


def _extract_features(image_array: np.ndarray) -> np.ndarray:
    """
    Compresses raw multidimensional pixel data into a 21-element global statistical footprint.

    Rather than feeding thousands of raw pixels directly into a machine learning model,
    this function compresses each image into a low-dimensional summary vector containing
    basic statistics (means, standard deviations, and percentiles) for each color channel.
    """
    # Normalize pixel intensity values from standard byte scale [0, 255] to decimals [0.0, 1.0].
    # This prevents scaling biases between different image types.
    image_array = image_array.astype(np.float32) / 255.0
    
    if image_array.ndim == 2:
        image_array = np.stack([image_array, image_array, image_array], axis=-1)

    if image_array.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image_array.shape}")

    # Flatten the 2D spatial dimensions (Height, Width) into a 1D list of pixels.
    # This reshapes the array from (Height, Width, 3 Channels) to (Total Pixels, 3 Channels).
    pixels = image_array.reshape(-1, image_array.shape[-1])
    
    # Extract structural color indicators across each of the 3 channels:
    # 1. Channel Means (3 values): Represents the overall brightness level of each channel.
    band_means = pixels.mean(axis=0)
    
    # 2. Channel Standard Deviations (3 values): Represents color contrast and texture variability.
    band_stds = pixels.std(axis=0)
    
    # 3. Percentiles (15 values): Evaluates lighting and structural distributions.
    # We calculate the 5th, 25th, 50th (median), 75th, and 95th percentiles across all 3 bands.
    # The output is reshaped from a (5, 3) matrix to a flat 15-element array.
    percentiles = np.percentile(pixels, [5, 25, 50, 75, 95], axis=0).reshape(-1)
    
    # Stitch the calculated arrays together: 3 (means) + 3 (stds) + 15 (percentiles) = 21 features.
    return np.concatenate([band_means, band_stds, percentiles]).astype(np.float32)


def _resolve_image_path(image_value: object, base_dir: Path) -> Path:
    """
    Resolves relative or absolute image filenames against candidate data subdirectories.

    Because index sheets (like CSV tables) might refer to images using absolute paths, relative paths,
    or paths nested in distinct directories, this function sequentially probes candidate locations
    to make file-system lookups highly robust.
    """
    if isinstance(image_value, (str, os.PathLike)):
        candidate = Path(str(image_value))
        
        # If the index already defines an absolute path, use it directly.
        if candidate.is_absolute():
            return candidate
        
        # Iteratively search through common directory layout variations:
        # 1. Directly under the base directory (e.g., base_dir/image_name.png)
        # 2. Base directory + relative subpath from CSV (e.g., base_dir/images/image_name.png)
        # 3. Under an explicit 'images' subfolder (e.g., base_dir/images/image_name.png)
        # 4. Strip nested paths to find the filename directly under 'images' (e.g., base_dir/images/image_name.png)
        for probe in [base_dir / candidate, base_dir / candidate.name, base_dir / "images" / candidate, base_dir / "images" / candidate.name]:
            if probe.exists():
                return probe
                
        # If no probe succeeds, return the default absolute relative path to let standard OS errors trigger
        return base_dir / candidate
    raise TypeError(f"Unsupported image reference type: {type(image_value)!r}")


def _normalize_label(value: object) -> Optional[str]:
    """
    Maps varying numerical indices or text labels into standardized target categories.

    Different datasets describe bloom severity using different schemas (e.g., integers 1-3, strings like "Low Severity").
    This standardizes all annotations into exactly three classes: 'Low', 'Moderate', or 'High'.
    """
    if value is None:
        return None
        
    # Handle numeric encodings (e.g., standard remote-sensing database classes: 1 -> Low, 2 -> Moderate, 3 -> High)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if int(value) == 1:
            return "Low"
        if int(value) == 2:
            return "Moderate"
        if int(value) == 3:
            return "High"
        return str(int(value))
        
    # Handle string variations by stripping whitespace and mapping synonyms
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "low", "low_severity", "low severity"}:
            return "Low"
        if text in {"2", "moderate", "moderate_severity", "moderate severity"}:
            return "Moderate"
        if text in {"3", "high", "high_severity", "high severity"}:
            return "High"
        if text in {"0", "none"}:
            # Map null levels or zero severities to Low
            return "Low"
        return value.strip()
    return str(value)


def _load_summary_frame(data_root: Path) -> Optional[pd.DataFrame]:
    """
    Scans a directory for any present tabular catalog or manifest files.

    It searches for common naming patterns for central indexes (CSV or Excel) to determine
    if the directory contains structured dataset metadata.
    """
    # Order of priority for CSV format metadata indexes
    summary_candidates = [
        data_root / "dataset_summary.csv",
        data_root / "dataset_summary_256x256pixels.csv",
        data_root / "metadata.csv",
        data_root / "labels.csv",
        data_root / "train.csv",
        data_root / "data.csv",
    ]
    for candidate in summary_candidates:
        if candidate.exists():
            return pd.read_csv(candidate)

    # Secondary check: search for Excel spreadsheet variations
    for excel_name in ["dataset_summary.xlsx", "dataset_summary_256x256pixels.xlsx"]:
        excel_path = data_root / excel_name
        if excel_path.exists():
            try:
                return pd.read_excel(excel_path)
            except Exception:
                continue
    return None


def _discover_folder_samples(data_root: Path, limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Fallback crawler that infers sample records by scanning unstructured folders directly.

    Used when no central CSV or Excel metadata manifest is found. It recursively scans folders,
    identifies image samples, reads target labels from nested 'metadata.json' files, and builds
    a complete training dataset dynamically.
    """
    sample_dirs = []
    # Find directories inside the root folder, ignoring system folders or dotfiles (e.g., .ipynb_checkpoints)
    for path in sorted(data_root.rglob("*")):
        if not path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        
        # Confirm directory actually contains image files or metadata
        image_files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif"}]
        if image_files or (path / "metadata.json").exists():
            sample_dirs.append(path)

    if not sample_dirs:
        raise FileNotFoundError(f"No sample folders with image files were found under {data_root}")

    # Enforce performance capping to prevent loading huge folders
    if limit is not None:
        sample_dirs = sample_dirs[:limit]

    image_rows = []
    features = []
    labels = []
    
    for sample_dir in sample_dirs:
        # Step 1: Look for a localized JSON descriptor file
        metadata_payload = None
        metadata_path = sample_dir / "metadata.json"
        if metadata_path.exists():
            try:
                metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata_payload = None

        # Step 2: Resolve the best image channel representing actual spectral bands.
        # We prioritize raw visual images over segmented prediction maps or masks.
        preferred_names = ["visual_raw", "B04_raw", "B03_raw", "B02_raw", "B01_raw"]
        image_path = None
        for name in preferred_names:
            for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                candidate = sample_dir / f"{name}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            if image_path is not None:
                break

        # Fallback if preferred spectral names are not found: take the first valid image file
        if image_path is None:
            for candidate in sorted(sample_dir.iterdir()):
                if candidate.is_file() and candidate.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif"}:
                    if "prediction_map" not in candidate.name.lower():
                        image_path = candidate
                        break

        if image_path is None:
            continue

        # Step 3: Extract the label class. 
        # Check standard environmental properties inside 'metadata.json' first.
        label = None
        if metadata_payload is not None:
            for key in ["indicative_class", "severity", "label", "class", "class_name", "severity_level", "abun_class"]:
                if key in metadata_payload:
                    label = _normalize_label(metadata_payload[key])
                    if label is not None:
                        break

        if label is None:
            for key in ["label", "severity", "class", "class_name"]:
                if key in metadata_payload or metadata_payload is None:
                    continue

        # If metadata is missing, try to infer the severity label from the folder's name
        if label is None:
            text = sample_dir.name.lower()
            if "high" in text:
                label = "High"
            elif "moderate" in text:
                label = "Moderate"
            elif "low" in text:
                label = "Low"

        # Step 4: Save record outputs. Store the sample configuration metadata to trace file sources.
        row = {"uid": sample_dir.name, "label": label, "image_path": str(image_path)}
        if metadata_payload is not None:
            for key, value in metadata_payload.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    row[key] = value
                    
        image_rows.append(row)
        features.append(_extract_features(_read_image_array(image_path)))
        labels.append(label if label is not None else "unknown")

    if not features:
        raise FileNotFoundError(f"No readable image samples were found under {data_root}")

    # Stack separate 1D features vertically into an (N, 21) array and return them
    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels)
    metadata = pd.DataFrame(image_rows)
    return X, y, metadata


def load_dataset(
    dataset_name: str = "kostaspic/amfitrite-inland-waters-hab-sentinel2",
    data_dir: Optional[str] = None,
    label_column: str = "severity",
    image_column: str = "image",
    split: str = "train",
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load HAB images and labels from a local folder or a Hugging Face dataset.

    The loader first checks local folders such as ./data, ./datasets/<dataset_name>,
    and the provided data_dir. If the dataset is not present locally, it attempts to
    load it through Hugging Face datasets if the dependency is installed.
    """
    # -------------------------------------------------------------------------
    # LOCAL LOOKUP & DIRECTORY RESOLUTION
    # Determine the search path array based on relative, system, or default directories.
    # -------------------------------------------------------------------------
    base_dir = Path(data_dir).expanduser() if data_dir else None
    candidate_dirs = []
    if base_dir is not None:
        candidate_dirs.append(base_dir)
    candidate_dirs.extend([
        Path.cwd() / "data",
        Path.cwd() / "datasets" / dataset_name,
        Path.cwd() / dataset_name,
        Path.cwd().parent / dataset_name,
    ])
    candidate_dirs = [p for p in candidate_dirs if p.exists()]

    data_root = candidate_dirs[0] if candidate_dirs else None
    if data_root is not None:
        # Some structured archives extract directories nested in a '/data' folder
        if (data_root / "data").exists() and data_root.name != "data":
            data_root = data_root / "data"

        # -------------------------------------------------------------------------
        # STRATEGY 1: Structured folder catalog layout (e.g. 'dataset_summary.csv')
        # Each unique sample sits in its own subdirectory containing isolated band images.
        # -------------------------------------------------------------------------
        summary_df = _load_summary_frame(data_root)
        if summary_df is not None:
            # Find the best column candidates representing our labels (severity indices or abundance labels)
            label_candidates = [c for c in summary_df.columns if c.lower() in {"severity", "label", "target", "class", "class_name", "indicative_class", "abun_class", "new_indicative_class"}]
            if label_candidates:
                uid_candidates = [c for c in summary_df.columns if c.lower() == "uid"]
                if uid_candidates:
                    uid_col = uid_candidates[0]
                    label_col = label_candidates[0]
                    rows = []
                    features = []
                    labels = []
                    
                    # Parse image paths nested inside the sample's subfolder
                    for _, row in summary_df.iterrows():
                        uid_value = row[uid_col]
                        sample_dir = data_root / str(uid_value)
                        if not sample_dir.exists() or not sample_dir.is_dir():
                            continue
                            
                        # Search for the target satellite band to read features from
                        image_path = None
                        for name in ["visual_raw", "B04_raw", "B03_raw", "B02_raw", "B01_raw"]:
                            for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                                candidate = sample_dir / f"{name}{ext}"
                                if candidate.exists():
                                    image_path = candidate
                                    break
                            if image_path is not None:
                                break
                                
                        if image_path is None:
                            continue
                            
                        label = _normalize_label(row[label_col])
                        rows.append({"uid": str(uid_value), "label": label, "image_path": str(image_path)})
                        features.append(_extract_features(_read_image_array(image_path)))
                        labels.append(label if label is not None else "unknown")
                        
                        if limit is not None and len(rows) >= limit:
                            break
                            
                    if rows:
                        X = np.vstack(features).astype(np.float32)
                        y = np.asarray(labels)
                        metadata = pd.DataFrame(rows)
                        return X, y, metadata

        # -------------------------------------------------------------------------
        # STRATEGY 2: Single-level directory layout with a flat index table (e.g. 'labels.csv')
        # Here, images sit in one folder and are mapped directly by filenames.
        # -------------------------------------------------------------------------
        if (data_root / "labels.csv").exists() or (data_root / "metadata.csv").exists() or (data_root / "train.csv").exists() or (data_root / "data.csv").exists():
            labels_df = None
            for csv_name in ["labels.csv", "metadata.csv", "train.csv", "data.csv"]:
                csv_path = data_root / csv_name
                if csv_path.exists():
                    labels_df = pd.read_csv(csv_path)
                    break

            if labels_df is None:
                raise FileNotFoundError(
                    f"No label file found in {data_root}. Expected one of labels.csv, metadata.csv, train.csv, or data.csv."
                )

            # Identify target label columns and image filename references
            label_candidates = [c for c in labels_df.columns if c.lower() in {"severity", "label", "target", "class", "class_name"}]
            image_candidates = [c for c in labels_df.columns if c.lower() in {"image", "image_path", "filepath", "path", "filename", "file_name", "img"}]
            if not label_candidates:
                raise KeyError(f"No label column found. Available columns: {list(labels_df.columns)}")
            if not image_candidates:
                raise KeyError(f"No image column found. Available columns: {list(labels_df.columns)}")

            label_column = label_candidates[0] if label_column == "severity" else label_column
            image_column = image_candidates[0] if image_column == "image" else image_column

            features = []
            labels = []
            for _, row in labels_df.iterrows():
                image_path = _resolve_image_path(row[image_column], data_root)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")
                features.append(_extract_features(_read_image_array(image_path)))
                labels.append(row[label_column])

            X = np.vstack(features).astype(np.float32)
            y = np.asarray(labels)
            metadata = labels_df.copy()
            # Generate absolute track references within metadata
            metadata["image_path"] = [str(_resolve_image_path(row[image_column], data_root)) for _, row in labels_df.iterrows()]
            return X, y, metadata

        # -------------------------------------------------------------------------
        # STRATEGY 3: Fallback unstructured directory crawler
        # Used when no metadata indexes or layout structures exist at all.
        # -------------------------------------------------------------------------
        return _discover_folder_samples(data_root, limit=limit)

    # -------------------------------------------------------------------------
    # STRATEGY 4: REMOTE FALLBACK STREAMING
    # This strategy triggers if no local datasets are discovered.
    # It attempts to load datasets from the online Hugging Face Hub.
    # -------------------------------------------------------------------------
    try:
        from datasets import load_dataset as hf_load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"The dataset {dataset_name!r} is not available locally and the Hugging Face datasets package is not installed."
        ) from exc

    warnings.warn("Loading from Hugging Face requires the datasets package and network access.")
    dataset = hf_load_dataset(dataset_name, split=split)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    frame = dataset.to_pandas()

    label_candidates = [c for c in frame.columns if c.lower() in {"severity", "label", "target", "class", "class_name", "severity_level"}]
    image_candidates = [c for c in frame.columns if c.lower() in {"image", "img", "filepath", "path", "filename", "file_name", "image_path"}]
    if not label_candidates:
        raise KeyError(f"Expected a label column in the Hugging Face dataset, got: {list(frame.columns)}")
    if not image_candidates:
        raise KeyError(f"Expected an image column in the Hugging Face dataset, got: {list(frame.columns)}")

    features = []
    labels = []
    
    # Process streamed representations of imagery (bytes, path links, arrays, PIL structures)
    for _, row in frame.iterrows():
        image_value = row[image_candidates[0]]
        
        # Scenario A: Streamed image metadata payload dictionary containing bytes or path references
        if isinstance(image_value, dict):
            if "bytes" in image_value:
                image_bytes = image_value["bytes"]
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                features.append(_extract_features(np.array(image, dtype=np.float32)))
            elif "path" in image_value:
                image_path = image_value["path"]
                image = Image.open(image_path).convert("RGB")
                features.append(_extract_features(np.array(image, dtype=np.float32)))
            else:
                raise TypeError(f"Unsupported image payload from Hugging Face: {image_value}")
                
        # Scenario B: Local temporary downloads containing direct string path links
        elif isinstance(image_value, (str, os.PathLike)):
            image = Image.open(image_value).convert("RGB")
            features.append(_extract_features(np.array(image, dtype=np.float32)))
            
        # Scenario C: Streamed numeric NumPy matrices
        elif isinstance(image_value, np.ndarray):
            image = Image.fromarray(np.uint8(image_value)).convert("RGB")
            features.append(_extract_features(np.array(image, dtype=np.float32)))
            
        # Scenario D: Raw PIL Image structures streamed directly over-the-air
        elif hasattr(image_value, "mode") and hasattr(image_value, "size"):
            image = image_value.convert("RGB")
            features.append(_extract_features(np.array(image, dtype=np.float32)))
        else:
            raise TypeError(f"Unsupported image type: {type(image_value)!r}")
            
        labels.append(row[label_candidates[0]])

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels)
    return X, y, frame