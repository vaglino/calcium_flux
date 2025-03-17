# Cell Calcium Imaging Analysis

A Python-based analysis pipeline for tracking cellular calcium dynamics using ratiometric imaging with GCaMP and RFP fluorescent markers.

## Overview

This software package processes and analyzes time-lapse microscopy data of B cells stained with calcium-sensitive GCaMP and RFP markers. The pipeline handles segmentation, tracking, and quantitative analysis of cellular calcium responses.

The workflow includes:

1. Segmentation of cells from ND2 image stacks
2. Tracking cells across frames
3. Analysis of fluorescence signals (GCaMP, RFP, and their ratio)
4. Normalization and alignment of data
5. Calculation of quantitative metrics
6. Visualization and export of results

## Features

- Automated cell segmentation from ND2 fluorescence microscopy files
- Cell tracking across time-series images
- Photobleaching correction
- Interactive landing frame detection and validation
- Signal normalization and alignment
- Statistical analysis of calcium dynamics including:
  - Area Under Curve (AUC)
  - Peak amplitude
  - Time to peak
- Comprehensive visualization tools
- Data export for further analysis

## Requirements

### Python Dependencies

```
numpy
pandas
matplotlib
scipy
scikit-image
opencv-python (cv2)
nd2
trackpy
tifffile
```

## File Structure

- `stack2tracks_nd2.py`: Processes ND2 image stacks, segments cells, and extracts parameters
- `segmentation.py`: Contains algorithms for cell segmentation
- `cell_parameters.py`: Functions to extract cell parameters from segmented images
- `analyze_experiment_on_surface_v06.py`/`v07.py`: Core analysis functions for tracking and processing
- `analysis_script_v03.py`: Main script for batch processing multiple image stacks
- `export_MFI_values.py`: Exports processed mean fluorescence intensity data

## Quick Start

1. Set up your directory structure with a `data` folder containing ND2 files
2. Run the main analysis script:

```python
from analysis_script_v03 import main
main()
```

Or alternatively, run directly:

```
python analysis_script_v03.py
```

## Workflow Details

### 1. Image Processing and Segmentation

The pipeline uses the RFP channel for segmentation since it's more stable than the calcium-sensitive GCaMP channel. The segmentation workflow includes:

- Preprocessing with adaptive histogram equalization and median filtering
- Binarization with adaptive thresholding
- Morphological operations to clean binary images
- Watershed segmentation to separate touching cells

### 2. Cell Tracking

Cells are tracked using the `trackpy` library, which implements the Crocker-Grier algorithm. The tracking considers:

- Maximum linking distance between consecutive frames
- Gap closing to handle temporary disappearances
- Filtering of short tracks

### 3. Signal Processing

- Double exponential correction for photobleaching
- Smoothing via uniform filtering
- Automatic landing frame detection with manual verification
- Normalization relative to landing point

### 4. Analysis and Statistics

The pipeline calculates several key metrics:

- Mean normalized GCaMP, RFP, and ratio signals
- Standard error of the mean (SEM)
- Area under curve (AUC) for calcium response
- Peak amplitude
- Time to peak

### 5. Interactive Features

The software includes an interactive interface for:

- Manual verification of landing frames
- Track inspection and filtering
- Visualization of segmentation and tracking results

## Output

The pipeline generates several output files:

- `*_segmentation.tif`: Segmentation results
- `*_tracks.csv`: Complete tracking data
- `*_tracks_raw.csv`: Raw tracking data before processing
- `*_tracks_clean.csv`: Processed tracks before alignment
- `*_tracks_align.csv`: Aligned and normalized tracks
- `*_mfi_data.csv`: Mean fluorescence intensity data for each condition
- `statistics_values.csv`: Summary statistics for each condition

## Advanced Usage

### Photobleaching Correction

You can customize photobleaching correction by providing parameters for the double exponential model:

```python
points, tracks, tracks_raw, tracks_clean, tracks_align = analyze_experiment_on_surface(
    image_stack, 
    threshold,
    gcamp_bleach_params=(0.5, 0.001, 0.0001), 
    rfp_bleach_params=(0.5, 0.0005, 0.00005)
)
```

### Customizing Track Processing

Filter tracks based on different criteria:

```python
# Require tracks to be at least 200 frames long
tracks = tp.filter_stubs(tracks, 200)

# Filter out tracks that appear after frame 10
tracks = tracks.groupby('particle').filter(lambda x: x['frame'].min() < 10)
```

## Troubleshooting

- If segmentation quality is poor, try adjusting the `threshold` parameter
- For cell tracking issues, adjust `max_linking_distance` and `max_gap_closing` parameters
- If landing frame detection is inaccurate, use the interactive interface to manually adjust

## License

[MIT License](LICENSE)

## Acknowledgements

Developed for the analysis of B cell calcium signaling in response to surface-bound antigens.
