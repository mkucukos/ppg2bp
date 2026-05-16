# Blood Pressure Estimation from PPG Signals

A deep learning pipeline that estimates systolic (SBP) and diastolic (DBP) blood pressure from Photoplethysmography (PPG) signals using a three-channel CNN trained on peak-by-peak beat windows.

## Table of Contents

- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Stage 1: Data Preparation & Preprocessing](#stage-1-data-preparation--preprocessing)
- [Stage 2: Model Training, Evaluation & Analysis](#stage-2-model-training-evaluation--analysis)
- [Data Organization](#data-organization)
- [How to Reproduce](#how-to-reproduce)
- [Notes](#notes)

## Project Structure

```
├── segment_to_cycle_loader.ipynb              # Stage 1: Data preprocessing
├── three_channel_ppg_peak2peak_subjects.ipynb # Stage 2: Model training & analysis
├── data/
│   ├── dataframes/                            # Processed DataFrames from Stage 1
│   │   └── df_subject_5_segment_5_row_8721_peak_by_peak_120_sampled.pkl
│   └── preprocessing_ready_set/              # Pre-processed datasets for direct use
│       └── about.txt
├── toy_segments/                              # Sample .pkl files for quick testing
├── models/                                   # Saved PyTorch model checkpoints
├── results/                                  # Evaluation outputs and predictions
├── requirements.txt
└── README.md
```

## Pipeline Overview

1. **Stage 1** — `segment_to_cycle_loader.ipynb`: raw PPG/ABP segments → filtered, quality-checked, beat-extracted DataFrame
2. **Stage 2** — `three_channel_ppg_peak2peak_subjects.ipynb`: preprocessed DataFrame → CNN training, evaluation, and analysis

---

## Stage 1: Data Preparation & Preprocessing

**Notebook:** `segment_to_cycle_loader.ipynb`

### 1.1. Segment Loading
- **Input:** Segmented PPG/ABP data from `.pkl` files, organised by subject and segment
- **Function:** `load_segments_from_directory`
- **Operations:** apply duration filters (2–10 min), optimise dtypes for memory, merge into `segments_by_subject_merged`

### 1.2. Signal Filtering
- **Method:** Butterworth bandpass filter (0.5–12 Hz)
- **Function:** `apply_filter_to_segments`
- Filtered PPG replaces the original signal in place

### 1.3. Feature Extraction & HRV Analysis
- **Tool:** NeuroKit2
- **Features:** PPG peaks, heart rate, HRV metrics, quality scores
- Segments are extended with a `data` DataFrame and an `info` metadata dict

### 1.4. Quality Filtering
- **Metric:** Mean `PPG_Quality` score
- **Threshold:** 0.92 (configurable)
- Segments below threshold are dropped

### 1.5. RR Interval Validation
- **Criteria:** 0.4 s ≤ RR ≤ 1.5 s with ≥ 80% valid intervals
- **Output:** `cleaned_segments_by_subject`

### 1.6. Bottom Detection
- Identifies valley indices in PPG waveform between peaks; stored in `info`

### 1.7. Visualisation
- Plots ABP and PPG signals with marked peaks and bottoms for visual QA

### 1.8. Beat Extraction
- **Function:** `extract_beats_with_raw_and_norm`
- Extracts peak-to-peak windows, resamples PPG to 120 samples, extracts SBP (max) and DBP (min) from the corresponding ABP window
- **Output columns:** `ppg_norm_120`, `ppg_raw_120`, `sbp`, `dbp`, `segment_id`, `abp_raw`

### 1.9. Data Persistence
- Saved as a `.pkl` file in `data/dataframes/` with an encoded filename

---

## Stage 2: Model Training, Evaluation & Analysis

**Notebook:** `three_channel_ppg_peak2peak_subjects.ipynb`

### 2.1. Data Loading
- Load processed DataFrame from Stage 1 (single file or concatenation of multiple)

### 2.2. Outlier Filtering
- Remove rows with mean ABP outside a specified confidence interval

### 2.3. Per-Subject Trimming
- Fix a target number of windows per subject (e.g., 1000–1001) for balanced representation

### 2.4. Blood Pressure Categorisation
- Classify beats into Normal / Elevated / Stage 1 / Stage 2 using custom rules
- Visualise class balance

### 2.5. Data Splitting
- **Strategy:** Subject-wise splits to prevent data leakage
- **Splits:** Train / Validation / Test

### 2.6. Data Preparation
- Arrays: `ppg_train/val/test` (PPG windows), `abp_train/val/test` ([SBP, DBP] pairs)
- Remove NaN values; shuffle with fixed seeds while maintaining alignment

### 2.7. Three-Channel PPG Representation
| Channel | Description |
|---|---|
| PPG | Original signal |
| VPG | First derivative (Velocity PPG) |
| APG | Second derivative (Acceleration PPG) |

- **Output shape:** `(N, 3, 120)`

### 2.8. Normalisation
- Z-score per channel, applied independently

### 2.9. PyTorch Integration
- Custom `PPGABPDataset` class wrapping the arrays
- `DataLoader` for efficient batching

### 2.10. Model Architecture
```
PPGtoABPRegressor
├── Input: (3, 120) three-channel PPG tensor
├── Conv1D layers with BatchNorm and ReLU
├── Dropout for regularisation
├── Flatten → Linear layers
└── Output: 2 values (SBP, DBP)
```

### 2.11. Training Configuration
- **Optimiser:** Adam
- **Loss:** MAE (L1 Loss)
- Training and validation loss tracked per epoch

### 2.12. Evaluation Metrics
- MAE distribution histograms
- Bland–Altman plots (MAP, SBP, DBP)
- Scatter plots (predicted vs. true)
- Optional global linear calibration for bias correction

### 2.13. Output Management
- Models saved to `models/`; results saved to `results/`

---

## Data Organisation

### `data/dataframes/`
Processed DataFrames generated by Stage 1.

- **Format:** `.pkl`
- **Naming pattern:** `df_subject_{N}_segment_{M}_row_{R}_peak_by_peak_120_sampled.pkl`
  - `N`: number of unique subjects, `M`: number of segments, `R`: total beat windows
- **Example included:** `df_subject_5_segment_5_row_8721_peak_by_peak_120_sampled.pkl`

### `data/preprocessing_ready_set/`
Pre-processed datasets ready for immediate use in Stage 2, bypassing the time-intensive Stage 1 pipeline.

The full 137-subject dataset (`df_137_sampled_peak_by_peak_rows_fixed_1000.pkl`) — 137 subjects × 1000 beat windows each — is not included in this repo. Contact the author to obtain it.

### `toy_segments/`
40 sample `.pkl` segment files drawn from `saved_subjects_32`, suitable for end-to-end testing of Stage 1 without downloading the full raw dataset.

---

## How to Reproduce

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

```bash
pip install -r requirements.txt
```

### Option A: Quick Start (toy dataframe — 5 subjects)

Load the included pre-processed dataframe and go straight to Stage 2:

```python
import pandas as pd
df = pd.read_pickle('data/dataframes/df_subject_5_segment_5_row_8721_peak_by_peak_120_sampled.pkl')
```

```bash
jupyter notebook three_channel_ppg_peak2peak_subjects.ipynb
```

### Option B: Full Pipeline (toy segments → Stage 1 → Stage 2)

1. Place segment `.pkl` files in the appropriate directory (or use `toy_segments/`)
2. Run Stage 1:
   ```bash
   jupyter notebook segment_to_cycle_loader.ipynb
   ```
   Outputs a processed DataFrame to `data/dataframes/`
3. Run Stage 2:
   ```bash
   jupyter notebook three_channel_ppg_peak2peak_subjects.ipynb
   ```

Check `models/` for saved checkpoints and `results/` for predictions and analysis.

---

## Notes

- All processing maintains **subject-wise separation** to prevent data leakage
- **Reproducibility** is ensured through fixed random seeds throughout
- Quality thresholds, window lengths, and model architecture are configurable
- For large datasets, use a CUDA-enabled GPU and adjust batch sizes to available memory

## Contact

For questions or to request the full 137-subject dataset: murat.kucukosmanoglu@dprime.ai
