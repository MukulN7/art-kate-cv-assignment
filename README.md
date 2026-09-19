# Artikate CV / ML Engineer Assignment

A computer vision project for two-class object detection (bottle and cup) using the YOLO model family.

## Project Overview
- **Classes**:
  - `0`: bottle
  - `1`: cup
- **Raw Image Count**: 74
- **Current Status**: Raw dataset collected; reproducible train/validation split completed; annotation and training not started.

## Dataset & Train/Validation Split
- **Total Source Images**: 74
- **Split Ratio**: Approximately 80% train / 20% validation (59 train, 15 validation)
- **Random Seed**: `42` (reproducible)
- **Raw Dataset Integrity**: Raw images in `data/raw/` remain completely untouched.
- **Augmentation**: NOT performed.
- **Annotation**: NOT performed yet.

## Roadmap
1. Dataset split (Completed)
2. Annotation
3. YOLO training
4. Evaluation
5. ONNX export
6. Quantisation
7. Benchmarking
8. Failure analysis
