---
license: cc-by-4.0
task_categories:
- image-classification
language:
- en
tags:
- earth-observation
- remote-sensing
- sentinel-2
- water-quality
- harmful-algal-blooms
- environmental
pretty_name: Amfitrite Inland Waters Harmful Algal Bloom (HAB)
size_categories:
- 1K<n<10K
---

# Dataset Card for Amfitrite-Inland-Waters-HAB-Sentinel2

This dataset contains multispectral Sentinel-2 satellite imagery tiles focused on inland water bodies, classified by the severity of Harmful Algal Blooms (HABs). 
It is designed to train Deep Learning models (like CNNs) for environmental monitoring.

## Dataset Details

### Dataset Description

**Amfitrite-Inland-Waters-HAB-Sentinel2** is a specialized dataset designed for the detection and classification of Harmful Algal Blooms (HABs) in inland water bodies (lakes, reservoirs, and rivers).
The dataset consists of multispectral image tiles derived from **Sentinel-2** satellite imagery. The **CyFi** tool was used to predict the class of the HAB in selected pixels labeled as water by the Sentinel-2 L2A Scene Classification Map (SCL band). 
These predictions in combination with the Cyanobacteria Aggregated Manual Labels (CAML) cyanobacteria abundance dataset were used to generate an indicative classes namely “High”, “Moderate” and “Low".
The dataset includes raw spectral bands, water masks and metadata with abundance metrics for 4698 such cases.

This work supports the **dAIEDGE** project's goal of enabling distributed, trustworthy and efficient AI at the edge.

- **Curated by:** Konstantinos Pikounis
- **Funded by:** [dAIEDGE Project](https://daiedge.eu/) (A Network of Excellence for Distributed, Trustworthy, Efficient and Scalable AI
at the Edge).
- **Shared by:** AMFITRITE Project
- **Language(s) (NLP):** English (Metadata)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### Dataset Sources

- **Repository:** [https://huggingface.co/datasets/kostaspic/amfitrite-inland-waters-hab-sentinel2](https://huggingface.co/datasets/kostaspic/amfitrite-inland-waters-hab-sentinel2)

## Uses

### Direct Use

- **AI Training:** Training Deep learning to detect HABs.
- **Edge AI Training:** Training lightweight Convolutional Neural Networks (CNNs) for deployment on edge devices to detect HABs in real-time.
- **Environmental Monitoring:** Developing algorithms to detect early-stage algal blooms in inland waters.

## Dataset Structure
The dataset is organized by unique identifiers (UIDs) which are a concatenation of a number randomly associated with a geographical area and the date of the satellite imagery. The folder names are these uids.

**Folder Structure per Sample:**
- **Raw Spectral Bands (`*_raw.tif`):** 12 spectral bands from Sentinel-2 (B01-B08, B8A, B09, B11, B12).
- **Scene Classification (`SCL_raw.tif`):** A map classifying pixels into 11 categories (e.g., water, clouds, vegetation, snow) based on the Sen2Cor processor.
- **AOT Map (`AOT_raw.tif`):** An Aerosol Optical Thickness map, based on the Sen2Cor processor.
- **Metadata (`metadata.json`):** A JSON file containing comprehensive details about the image acquisition, measurements and the model
predictions.
  - `item_id`: The unique Sentinel-2 product identifier (from ESA/Planetary Computer).
  - `per_clouds`: The percentage of the tile covered by clouds (SCL values: 3 - Cloud shadows, 7 - Unclassified, 8 - Cloud medium probability, 9 - Cloud high probability and 10 - Thin cirrus).
  - `water_pixels`: The total number of pixels classified as water (SCL class 6) in the tile.
  - `uid`: The unique Case ID matching the folder name (derived from CAML).
  - `abun`: The highest abundance values derived from CAML.
  - `tile_size_10m_pixels`: The desired width/height of the tile in pixels (always 365).
  - `date`: The date of the measurement  (YYYY-MM-DD) (derived from CAML).
  - `center_lat`: The latitude coordinates of the tile center.
  - `center_lon`: The longitude coordinates of the tile center.
  - `points_data`: A list of specific ground truth measurements associated with this location (derived from CAML). Each entry contains:
      - `uid`: The ID of the specific measurement point.
      - `lat` / `lon`: Exact location of the measurement.
      - `date`: Date of the in-situ measurement (may differ slightly from image date).
      - `abun`: The raw abundance value.
      - `severity`: The mapped severity class (1=Low, 2=Moderate, 3=High).
  - `High counts` / `Moderate counts` / `Low counts`: The number of pixels in the 10pixel grid (see **Data Collection and Processing** Section) classified into each severity level by the **CyFi** tool for this specific tile.
  - `compatibility_analysis`: Metadata generated during dataset curation to validate the match between satellite predictions and ground truth.
      - `clusters`: Groupings of ground truth points (if multiple measurements exist within the tile). Contains centroid coordinates `centroid_lat` / `centroid_lon`, the data / prediction compatibility score (see **Data Collection and Processing** Section) `cluster_incompatibility_score`, the uid of the most compatible measurement point `most_compatible_uid` as well as a list of all `points` with `uid`, `severity` derived from CAML and the respective `compatibility_score`.

- **CyFi Prediction (`cyfi_lattice_predictions.csv`):** A csv file with the **CyFi** prediction for the 10pixel grid points.
  - `sample_id`: “sample_0” + ascending number
  - `latitude`: the latitude of the point
  - `longitude`: the longitude of the point
  - `density_cells_per_ml`: **CyFi** prediction
  - `severity`: **CyFi** prediction
  - `pixel_row`: the row of the pixel in the tile
  - `pixel_col`: the column of the pixel in the tile.

- **CyFi Prediction Map (`cyfi_prediction_map.png`):** A composite visualization used for visual verification. 
It displays the Sentinel-2 visual (RGB) band overlaid with the **CyFi** prediction grid (color-coded points) and any available in-situ measurements (marked with an 'x' and color-coded by severity) to make comparison between model outputs and in situe mesurments easier.
An example image is shown below.
<img src="images_for_readme/433_2019-06-27-High.png" alt="CyFi Prediction Map Example" width="500"/>


**Dataset Summary File (`dataset_summary.xlsx`):** A tabular summary aggregating metadata, measurements and classification results for all cases in the dataset.
  - `uid`: The unique Case ID matching the folder name.
  - `uid_CAML`: A list of the original identifiers from the CAML dataset matching the location and date.
  - `case`: A numeric identifier associated with the specific geographical region/square.
  - `item_id`: The unique Sentinel-2 product identifier
  - `date`: The acquisition date (YYYY-MM-DD).
  - `lat` / `lon`: The latitude and longitude coordinates of the tile center.
  - `abun`: The list of the in-situ cyanobacteria abundance values associated with this case from CAML.
  - `abun_class`: The severity class derived from the abun highest value.
  - `per_clouds`: The percentage of the tile covered by clouds (Cloud medium probability, Cloud high probability, Thin cirrusm Cloud shadows and Unclassified pixels from SCL).
  - `water_pixels`: The total number of pixels classified as water in the tile.
  - `high_pred` / `mod_pred` / `low_pred`: The count of grid points classified as High, Moderate, and Low respectively by the CyFi tool.
  - `HAB_status`: The preliminary classification status derived solely based on `high_pred`, `mod_pred` and `low_pred` values.
  - `roughness_median`: A statistical metric representing the spatial variability of the prediction map, used to filter noisy data.
  - `compatible_severities`: The class of the points found compatible with the CyFi predictions in a radius 300m around them (see **Data Collection and Processing / 3. Imagery Classification Method** section)
  - `outlier_fraction`: The fraction of grid points considered statistical outliers, used for quality control.
  - `indicative_class`: The indicative label assigned to the tile (High, Moderate, Low).
  - `category`: Indicates if the case contains in situe measurments ("with_measurements") or is an augmented sample ("without_measurements").
  - `training_priority`: A numeric flag used to curate a diverse training set by reducing redundancy within specific cases. Value 1 denotes high-priority samples, including unique "mixed" class transitions and representative "pure" examples, while 2 indicates redundant samples similar to those already prioritized. By utilizing cases from both priorities, the final dataset is effectively enlarged to maximize training volume. 

**256x256 Dataset Summary File (`dataset_summary_256x256pixels.xlsx`):** A tabular file containing most of the tiles in Dataset Summary File (a few were excluded).  
  - `uid`, `uid_CAML`, `case`, `item_id`, `date`, `lat`, `lon`, `abun`,  `abun_class`, `per_clouds`, `category` and `training_priority`: same as in dataset_summary.xlsx
  - `crop_row` / `crop_col`: The row and column coordinates representing the upper-left corner of the optimal 256x256 extracted tile within the 365x365 matrix.
  - `new_water_pixels`: Total water pixels strictly within the 256x256 cropped area.
  - `new_high_pred` / `new_mod_pred` / `new_low_pred`: CyFi severity counts strictly within the 256x256 cropped area.
  - `new_low_percentage`: The percentage of low-severity predictions within the crop.
  - `new_indicative_class`: The recalculated, definitive label (High, Moderate, Low) for the 256x256 cropped tile to account for potential label shift.

### Tiling Strategy & Dimensions
**1. Custom Augmentation (365x365):** The dataset was created with the goal of supporting
**256x256** pixel input for CNNs.
- **Standard Dimensions:** Most tiles are cropped to
**365x365 pixels**.
- **Rationale:** This size was chosen because 256 * sqrt(2) ~ 362. Providing a 365px tile allows data augmentation pipelines to rotate the image around the center and extract a 256x256 crop without introducing empty corners (black padding).

**2. Direct Edge AI Training (256x256):** For users who require strict, pre-formatted 256x256 CNN inputs without custom augmentation, the dataset provides exact extraction coordinates (`crop_row`, `crop_col`) in the 256x256_dataset_summary Excel file. These coordinates define an optimal 256x256 crop. Users utilizing this approach must use the `new_indicative_class` label.

**- Exceptions:** Some tiles may have smaller dimensions if the original satellite swath ended near the target location or if the water body was near the edge of a tile.

## Dataset Creation

### Curation Rationale

Harmful Algal Blooms pose a significant threat to public health and ecosystems. This dataset was created as a means to train Deep Learning tools to identify HABs using satellite images, with a focus on light weight models compatible with edge devices onboard satellites for creating early warning systems. The annotation combines precise in-situ measurements (CAML) when available with prediction of dedicated ML tools (**CyFi**).

### Source Data
The source data consists of multispectral optical satellite imagery acquired by the **Sentinel-2** mission, which is part of the **Copernicus Programme** operated by the **European Space Agency (ESA)**.

#### Annotation Logic
The primary goal of this dataset is to train satellite imagery classifiers capable of identifying the presence of HABs (for monitoring and early warning purposes). Consequently, the labeling strategy follows a severity-priority principle:
- **High:** If any significant portion of the image contains a High severity bloom, the entire tile is categorized as "High".
- **Moderate:** If no High severity bloom is detected but a Moderate bloom exists (even in just a portion of the image), the entire tile is categorized as "Moderate".
- **Low:** The tile is labeled "Low" only if the water body is clear of High or Moderate blooms.
This approach ensures that the model is trained to flag potential environmental threats even if they are localized within a larger water body, rather than averaging the signal across the entire image.

#### Data Collection and Processing
1. **Location Sourcing**
Geographical locations and abundance measurements were sourced from the Cyanobacteria Aggregated Manual Labels (CAML) dataset. These measurement locations were clustered into 2560m x 2560m squares (equivalent to 256 x 256 pixels at 10m resolution). A unique ID was assigned to each square, forming the first part of each record's uid.
2. **Imagery Retrieval**
The pre-defined squares were used to query the Microsoft Planetary Computer for Sentinel-2 Level-2A imagery within a window of ±15 days relative to the CAML measurement dates. From this range, the image with the minimum cloud coverage was selected, prioritizing dates closest to the actual measurement. The imagery date comprises the second part of the uid. A cloud coverage threshold of < 7.5% was applied to maximize the pixels processable by the **CyFi** tool. Tiles of 365 x 365 pixels were extracted and aligned. Sentinel-2 bands with 20m and 60m resolutions were upsampled to 10m using bilinear interpolation via the rasterio library to ensure perfect spatial alignment across all bands. In the case of the SCL map the nearest neighbor resampling method was used. The dataset was further augmented with images of the preselected squares from dates not matching the original CAML measurement dates.
3. **Imagery Classification Method**
Initially, a grid was established every 10 pixels for all areas labeled as "water" in the Scene Classification Layer (SCL) map. These points were processed by the **CyFi** tool. The entire image was then classified based on this grid:
 **Low:** <10 points classified as “Moderate” or “High,” indicating high class purity.
 **Moderate:** Not classified as "Low" and containing < 10% points labeled as "High."
 **High:** > 10% of points classified as "High."
For cases with ground-truth measurements, **CyFi** predicted abundance for all pixels within a 300m radius of the measurement point. Assuming a log-normal distribution, the Z-score of the ground-truth measurement was computed against the weighted satellite-derived prediciton distribution. If the deviation exceeded 3σ, the **CyFi** prediction was favored, but the image was flagged for manual review. If multiple measurements fell within the same 300m radius, the Wasserstein Distance (WD) between the predicted distribution and the measurement-derived distribution was used to define compatibility. Cases where the normalized WD (divided by the predicted standard deviation) exceeded 3 were also favored toward **CyFi** and manually inspected. Additional manual checks or exclusions were performed for images exhibiting high spatial roughness or numerous outliers in the predicted grid, including images without associated ground-truth measurements.
5. **Final Selection**
Images were selected if they contained > 6554 pixels (~10% of the 256 x 256 core area) categorized as water in the SCL map and had at least 50 grid points successfully predicted by the **CyFi** tool.

#### Who are the source data producers?

-   **Satellite Imagery:** European Space Agency (ESA) Copernicus Sentinel-2 mission.
-   **Measurements:** Gupta et al. (NASA/DrivenData) via the CAML dataset.


### Annotation process / Who are the annotators?
Addressed in **Source Data / Data Collection and Processing** section above


#### Personal and Sensitive Information
This dataset contains satellite imagery of public water bodies and scientific metadata. It contains no personal identifiable information.

## Bias, Risks, and Limitations

-   **Geographic Bias:** The image locations were strictly collected based on the sampling points provided in the **Gupta et al. (2024) CAML dataset**. As such, the dataset inherits the geographic distribution of that study (North America sites). 
-   **Cloud Cover:** While filtered, some tiles may still contain thin cirrus clouds or cloud shadows that affect spectral response.
-   **Land Adjacency:** Inland water tiles often include shorelines. While an SCL mask is provided, "mixed pixels" at the water's edge can introduce noise.
-   **Inherited Model Biases:** The methodology incorporates inherent biases from the **CyFi** tool, including potential systematic errors derived from its training data distribution and the spectral constraints of the Sentinel-2 sensor.
-   **Taxonomic Specificity:** The **CAML** dataset focuses exclusively on Harmful Algal Blooms (HABs) induced by cyanobacteria. Consequently, the classification logic and abundance predictions developed in this study may not be generalizable to HABs caused by other microorganisms, **possessing** distinct optical properties and ecological signatures.


### Recommendations
**1. Critical Impact of Cropping on Class Labels and Content:** 
**This is a very important point.** 
Users must be aware that while the dataset provides ~365x365 pixel tiles to facilitate augmentation (rotation), performing a fixed center-crop to 256x256 pixels without validation introduces significant risks:
- **Risk of Region of Interest Loss:** Inland water bodies are spatially complex and irregular. In some cases, the water body may be located off-center within the 365x365 tile. A blind center-crop may result in a dataset sample dominated by land features, effectively excluding the water body entirely. **Users are strongly advised to verify the water mask (SCL) percentage after cropping.** Example: The red box indicates the 256x256 center crop. In this case, the crop excludes the water body, leaving only land. <img src="images_for_readme/2241_217-05-09-High.png" alt="Example of center crop excluding the water body" width="500"/>
- **Risk of Class Label Shift:** Algal blooms are often patchy and heterogeneous. The "Class" label provided in the metadata characterizes **the entire 365x365 tile**. If the pixels contributing to a specific class are located near the edges of the tile, a center-crop will remove them. Consequently, the remaining 256x256 image may effectively contain only abundance features from other classes, but it will still carry the original label. This introduces label noise. **Users should ideally re-calculate the class label based on the points remaining within their specific crop window.** Example: The original tile is classified as "High" due to the bloom on the bottom part. The center crop (red box) excludes these high-severity points. If the label is not updated, this cropped image will be incorrectly trained as "High" despite only showing "Moderate" characteristics. <img src="images_for_readme/61_2019-09-21-High.png" alt="Example of label shift due to cropping" width="500"/>
- **The Solution:** To completely mitigate these risks, users should utilize the optimal **cropping coordinates (`crop_row`, `crop_col`) corresponding to the pixel of the upper right corner of the 256x256 pixel tile** provided in the **dataset_summary_256x256pixels.xlsx**.
**. If utilizing these specific 256x256 crops, users **must** train their models using the `new_indicative_class` label, which has been strictly recalculated based only on the CyFi predictions that fall within that specific 256x256 cropped window. Additionally, those cases where the class of the 256x256 tile did not match that of the 365x365 tile were manually reviewed and reclassified if needed.

**2. Train models using Scene Classification Layer (SCL) mask:**
Users could utilize the provided Scene Classification Layer (SCL) to mask out non-water pixels during training to prevent the model from learning land features.

**3. Verify Dimensions:**
As noted in the structure section, some images may be smaller than 365x365. Users should verify dimensions before batch processing.
The following compact Python snippet can be used to check file shapes:

```python
import rasterio

# Check dimensions of a specific band
with rasterio.open("path_to_tiff/B04_raw.tif") as src:
   print(f"Shape: {src.width} x {src.height}")
```

## References

Gupta, S., Gelbart, E., Gupta, R., Wetstone, K., and Dorne, E. (2024). Cyanobacteria aggregated manual labels dataset (nasa and drivendata).
```bibtex
@misc{gupta_caml_2024,
  author    = {Gupta, S. and Gelbart, E. and Gupta, R. and Wetstone, K. and Dorne, E.},
  title     = {Cyanobacteria Aggregated Manual Labels Dataset (NASA and DrivenData)},
  year      = {2024},
  publisher = {NASA SeaBASS},
  doi       = {10.5067/SeaBASS/CAML/DATA001},
  url       = {http://dx.doi.org/10.5067/SeaBASS/CAML/DATA001}
}
```

Dorne, E., Wetstone, K., Cerquera, T. B., and Gupta, S. (2024). Cyanobacteria detection in small, inland water bodies with cyfi. In Proceedings of the 23rd Python in Science Conference, pages 154–173.
```bibtex
@inproceedings{dorne_cyfi_2024,
  author    = {Dorne, E. and Wetstone, K. and Cerquera, T. B. and Gupta, S.},
  title     = {Cyanobacteria detection in small, inland water bodies with CyFi},
  booktitle = {Proceedings of the 23rd Python in Science Conference},
  pages     = {154--173},
  year      = {2024},
  doi       = {10.25080/pdhk7238}
}
```

# Source Data Attribution
This dataset contains modified **Copernicus Sentinel data [2017-2021]**. Original data provided by the **European Space Agency (ESA)** via the **Microsoft Planetary Computer**. Use of Sentinel data is subject to the [ESA Legal Notice](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice).
