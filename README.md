# TRAFFIQ Autonomous Pipeline

This README documents the full working pipeline used in this project:
- Dataset source: `dataset_car/`
- Data collection: `scripts/collect_data_cv.py`
- CV preprocessing: `scripts/cv_pipeline.py`
- Training: `training/train_dave_cv.py`
- Inference: `inference/run.py` + `inference/decision.py`
- Dataset quality tools: `utils/check_dataset.py` and `utils/check_datacv.py`

It is written for your current workspace structure and current script behavior.

## 1) Pipeline Overview

The system is a hybrid driving stack:

1. Camera frame (`640x480` RGB)
2. OpenCV pipeline extracts:
   - white line position
   - obstacle detections
   - normalized/cropped CNN input
3. CNN predicts `[speed, direction]`
4. Decision layer combines CNN + CV signals
5. Final command is sent to the car

Conceptual flow:

```text
Camera Frame
   -> scripts/cv_pipeline.py (normalize, crop, line, obstacle, cnn_input)
   -> CNN model (training/train_dave_cv.py architecture)
   -> [speed, direction]
   -> inference/decision.py (safety + blending overrides)
   -> final [speed, direction] to vehicle control
```

## 2) Folder Responsibilities

- `dataset_car/`
  - Main training source in your current workflow.
  - Contains multiple sub-datasets (for example `american_steel_adam_2`, `athena_rainer_bosch`, etc.).

- `scripts/collect_data_cv.py`
  - Manual driving collector.
  - Captures frames and logs both steering/direction and speed/throttle.

- `scripts/cv_pipeline.py`
  - Shared image preprocessing and CV feature extraction.
  - Used by both training preprocessing and runtime/inference.

- `training/train_dave_cv.py`
  - Main trainer for TRAFFIQ v2 model.
  - Multi-output model predicts `[speed, direction]`.

- `inference/run.py`
  - Runtime loop (camera -> cv pipeline -> model -> decision -> control).
  - Includes watchdog safe-stop logic.

- `inference/decision.py`
  - Rule-based decision layer for safety and obstacle handling.

- `utils/check_dataset.py`
  - Basic dataset validation for steering-style labels.

- `utils/check_datacv.py`
  - v2-style utility script (label migration + dataset checks for direction/speed).

- `models/`
  - Stores trained `.h5`, exported `.tflite`, and training logs/plots.

## 3) Environment Setup

From the `traffiq/` folder:

```bash
python3 -m venv ../.venv
source ../.venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Current `requirements.txt` includes:
- `numpy`, `pandas`, `pillow`, `matplotlib`, `scikit-learn`, `tensorflow`, `pygame`, `opencv-python`

Notes:
- If running on Raspberry Pi for inference, lightweight `tflite_runtime` is often preferred.
- For simulator collection, install DonkeyCar simulator dependencies (`gym`, `gym_donkeycar`) if needed.

## 4) Data Collection Pipeline

Script: `scripts/collect_data_cv.py`

What it does:
1. Reads keyboard control (W/A/S/D + arrows)
2. Records frame + control labels while driving
3. Saves images and `labels.json` in timestamped `dataset/<session>/`

Controls:
- `W` / Up: increase speed
- `S` / Down: brake/decrease speed
- `A` / Left: steer left
- `D` / Right: steer right
- `R`: start/pause recording
- `Q`: quit and save manifest

Run:

```bash
python scripts/collect_data_cv.py
```

Collection best practices:
- Do many turns (left and right), not only straights.
- Vary speed deliberately (slow in turns/near obstacles, faster on clear straights).
- Avoid long stretches of near-identical frames.

## 5) Dataset Quality Checks

```bash
python utils/check_datacv.py migrate --data_dir dataset/<your_session>
```



## 6) CV Pipeline Details

Script: `scripts/cv_pipeline.py`

Main stages:

1. `normalize_lighting(image)`
   - LAB + CLAHE on L channel to reduce lighting variation.

2. `crop_frame(image)`
   - Removes top and bottom irrelevant areas.
   - Uses dynamic crop fractions (~25% top, ~15% bottom).

3. `detect_white_line(cropped)`
   - HSV threshold for white, morphology cleanup, ROI search.
   - Returns line found flag + normalized offset.

4. `detect_obstacles(cropped)`
   - Grayscale/adaptive threshold + white-line subtraction.
   - Contour filtering by area.
   - Returns nearest obstacle with side/size/proximity features.

5. `preprocess_for_cnn(cropped)`
   - Resize to `200x66`, convert RGB->YUV, normalize to `[0,1]` float32.

6. `run_pipeline(raw_frame)`
   - Runs all stages and returns:
     - `cnn_input`
     - `line`
     - `obstacles`
     - `debug_frame`

Standalone test mode exists in the script (`__main__`) for visual debugging.

## 7) Training Pipeline

Script: `training/train_dave_cv.py`

### 7.1 Inputs and labels

- Supports two dataset formats:
  - TRAFFIQ `labels.json`
  - DonkeyCar `catalog_*.catalog`
- Unified record fields are internally normalized to:
  - steering/direction
  - throttle/speed
  - absolute image path
  - source dataset name

### 7.2 Preprocessing and augmentation

Each batch sample:
1. Read image
2. Convert BGR->RGB
3. Clip controls to valid range
4. Optional augmentation (brightness, color shift, shadow, horizontal flip)
5. Run cv pipeline preprocessing
6. Output training pair `(image, [speed, direction])`

### 7.3 Rebalancing

`rebalance_direction_records()` reduces straight-driving bias and oversamples hard turns.

### 7.4 Model architecture

- DAVE-2 style shared CNN backbone
- Two heads:
  - speed head (`tanh`)
  - direction head (`tanh`)
- Concatenated final output order:
  - `[speed, direction]`

### 7.5 Loss and metrics

Custom weighted objective:
- speed component weight = `0.2`
- direction component weight = `0.8`
- direction uses Huber + MAE blend for robustness and precision

Metrics logged:
- `speed_mae`
- `direction_mae`

### 7.6 Outputs

Training produces:
- Best Keras model: `models/traffiq1_v2.h5`
- CSV logs: `models/training_log_v2.csv`
- Curves image: `models/training_curves_v2.png`
- INT8 TFLite model: `models/traffiq1_v2.tflite`

### 7.7 Run training

From `traffiq/`:

```bash
python training/train_dave_cv.py \
  --data_dir /home/remon/Documents/Trafiic_car_autonomous/traffiq/dataset_car
```

Optional useful flags:

```bash
# Disable fast defaults
python training/train_dave_cv.py --no_fast_mode

# Custom epochs
python training/train_dave_cv.py --epochs 100

# Explicit dataset subset
python training/train_dave_cv.py \
  --selected_datasets american_steel_adam_2 athena_rainer_bosch circuit_launch_ed_2

# Disable mixed precision
python training/train_dave_cv.py --no_mixed_precision
```

Run:

```bash
python inference/run.py --model models/traffiq1_v2.tflite --debug
```

Safety logic in decision layer (`inference/decision.py`):
- Blends CNN direction with line offset when lane is visible
- Holds previous direction and slows down when line is lost
- Reduces speed and applies avoidance steering on obstacle detection
- Emergency stop for large close center obstacles
- Clips outputs to `[-1, 1]`

## 9) Recommended End-to-End Workflow

1. Collect data with `scripts/collect_data_cv.py`.
2. Validate each session with `utils/check_datacv.py` (or `utils/check_dataset.py` for older format).
3. Move/organize curated sessions under `dataset_car/`.
4. Train with `training/train_dave_cv.py`.
5. Verify generated files in `models/`.
6. Deploy model and run on Pi/car runtime.

## 10) Quick Commands

```bash
# 1) Setup
source ../.venv/bin/activate
pip install -r requirements.txt

# 2) Collect
python scripts/collect_data_cv.py

# 3) Check
python utils/check_datacv.py check --data_dir dataset/<session>

# 4) Train (your current main dataset)
python training/train_dave_cv.py --data_dir /home/remon/Documents/Trafiic_car_autonomous/traffiq/dataset_car
# 5) Decision making
python inference/decision.py 
# 6) deployment
python inference/run.py

```

---
