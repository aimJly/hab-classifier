# HAB Severity Classification Project

This folder adds a runnable machine learning workflow for the Harmful Algal Bloom severity task using the repository’s satellite-imagery context.

## Structure
- src/hab_pipeline/data_loader.py: loads local image/label datasets or can switch to Hugging Face later
- src/hab_pipeline/train.py: trains a baseline severity classifier
- src/hab_pipeline/infer.py: predicts the severity for a single image
- tests/test_pipeline.py: smoke test for the full flow

## Usage

1. Install dependencies:
   pip install -r requirements.txt

2. Put your dataset in one of these locations:
   - ./data
   - ./datasets/kostaspic_amfitrite
   - ./kostaspic_amfitrite
   - or pass --data-dir /path/to/dataset

3. Your dataset should contain:
   - a CSV file with image references and a severity label column
   - image files referenced by the CSV

4. Train the model:
   python -m hab_pipeline.train --dataset-name kostaspic/amfitrite-inland-waters-hab-sentinel2 --data-dir /path/to/dataset --output-dir outputs

5. Predict a single image:
   python -m hab_pipeline.infer --image-path /path/to/image.png

## Notes
- The current baseline uses handcrafted image features with a Random Forest classifier.
- If you later upload the real dataset under a folder that contains a CSV and image files, the same pipeline will work without changing the code.
- If you want, the next step can be replacing the baseline with a CNN or transfer-learning model once the dataset is available.
