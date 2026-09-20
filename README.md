# Artikate CV / ML Engineer Assignment

## Video Walkthrough

[View the 6–8 minute submission walkthrough](https://drive.google.com/file/d/13fYavDQdnA3LUfrGoHqBvEU3lzmvC_cP/view?usp=sharing)

---

## 1. Overview
This project implements an end-to-end computer vision pipeline for two-class object detection targeting **bottle** (Class `0`) and **cup** (Class `1`). Built using the **YOLO11n** lightweight architecture, the project covers dataset preparation, manual YOLO bounding box annotation, model training, ONNX model export, FP16 precision reduction, model benchmarking, and failure mode analysis.

---

## 2. Repository Structure

```text
art-kate-cv-assignment/
├── .gitignore                      # Git ignore file for local envs and artifacts
├── README.md                       # Comprehensive project documentation (Part A5)
├── ANSWERS.md                      # Conceptual and system design responses (Parts B, C, D)
├── requirements.txt                # Python environment dependencies
├── data/                           # Dataset root directory
│   ├── data.yaml                   # YOLO dataset configuration file
│   ├── raw/                        # Original unedited captured images
│   ├── train/                      # Training split (59 images + labels)
│   │   ├── images/
│   │   └── labels/
│   └── val/                        # Validation split (15 images + labels)
│       ├── images/
│       └── labels/
├── models/                         # Serialized model checkpoints and exported ONNX models
│   ├── best.pt                     # Best PyTorch model checkpoint from training
│   ├── best_fp32.onnx              # Exported FP32 ONNX model (640x640)
│   └── best_fp16.onnx              # Reduced-precision FP16 ONNX model (640x640)
├── notebooks/                      # Google Colab execution notebook
│   └── model_training_artikate.ipynb
├── reports/                        # Evaluation plots and training artifacts
│   ├── training_results.csv        # Metrics log across all 50 training epochs
│   ├── training_curves.png         # Loss and metric evolution charts
│   ├── confusion_matrix.png        # Validation confusion matrix
│   ├── validation_predictions.jpg  # Ultralytics validation prediction visualization
│   └── validation_labels.jpg       # Ground-truth visualization
└── scripts/                        # Reproducible pipeline execution scripts
    ├── split_dataset.py            # Train/val dataset splitting script
    ├── train.py                    # Model training script
    ├── export_onnx.py              # ONNX model export script (FP32 & FP16)
    └── benchmark_onnx.py           # ONNX Runtime benchmarking & evaluation script
```

---

## 3. Dataset
- **Total Captured Images**: 74 original unedited images stored in `data/raw/`.
- **Target Classes**:
  - `0`: `bottle`
  - `1`: `cup`
- **Train / Validation Split**:
  - **Train**: 59 images (~80%)
  - **Validation**: 15 images (~20%)
  - Split performed deterministically using seed `42` prior to any augmentation.
- **Annotated Object Instance Counts**:
  - **Bottle (Class 0)**: 66 instances (52 train / 14 val)
  - **Cup (Class 1)**: 61 instances (46 train / 15 val)
  - **Total Bounding Boxes**: 127 instances
- **Dataset Variation & Realism**: Images feature realistic indoor office/desk environments, varying object orientations, background clutter, and lighting variations. Raw dataset images are preserved without destructive edits.

---

## 4. Annotation
- **Format**: Standard YOLO normalized format (`class_id x_center y_center width height`), with all coordinates scaled to `[0.0, 1.0]`.
- **Class Mapping**:
  - `0` $\rightarrow$ `bottle`
  - `1` $\rightarrow$ `cup`
- **Bounding Box Policy**: Bounding boxes tightly enclose the visible extent of each object instance. Partially occluded or cropped objects were annotated up to their visible boundaries.
- **Verification**: All 74 label files were validated programmatically to ensure exact 5-value line formatting, valid class IDs (`0` or `1`), and coordinate bounds within `[0, 1]`.

---

## 5. Training

### Training Configuration
Training was conducted on a **Google Colab Tesla T4 GPU**.

| Parameter | Value / Setting |
| :--- | :--- |
| **Model Architecture** | YOLO11n (`yolo11n.pt` pretrained base) |
| **Input Resolution** | $640 \times 640$ pixels |
| **Epochs** | 50 |
| **Batch Size** | 16 |
| **Optimizer** | `auto` (Ultralytics selected AdamW, $lr \approx 0.001667$, momentum 0.9) |
| **Random Seed** | 42 |
| **Patience** | 15 (Early stopping) |
| **Automatic Mixed Precision (AMP)** | Enabled (`amp=True`) |
| **Augmentations** | Ultralytics default online training augmentations |
| **Hardware** | Google Colab Tesla T4 GPU (CUDA `0`) |
| **Training Time** | $\approx 0.066$ hours ($\sim 4$ minutes) |

### Validation Metrics (PyTorch `best.pt`)

#### Overall Model Performance
| Metric | Value |
| :--- | :---: |
| **Precision** | 0.8538 |
| **Recall** | 0.6548 |
| **mAP@0.5** | 0.7867 |
| **mAP@0.5:0.95** | 0.5552 |

#### Per-Class Performance
| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: |
| **Bottle (Class 0)** | 0.7990 | 0.6430 | 0.7440 | 0.4870 |
| **Cup (Class 1)** | 0.9080 | 0.6670 | 0.8370 | 0.6290 |

*Note on Data Ingestion*: During training initialization, Ultralytics flagged some JPEG headers as corrupt and automatically restored them before proceeding. Training completed without loss of data.

---

## 6. ONNX Export
- **Export Tooling**: PyTorch to ONNX conversion via Ultralytics export module.
- **Opset Version**: 17
- **Input Geometry**: Static $1 \times 3 \times 640 \times 640$ NCHW tensor.
- **Execution Engine**: ONNX Runtime using `CUDAExecutionProvider` on Tesla T4.
- **Exported Formats**:
  1. **FP32 ONNX** (`best_fp32.onnx`): Full precision export.
  2. **FP16 ONNX** (`best_fp16.onnx`): Half-precision float16 export.
- **Precision Reduction Selection Rationale**: FP16 half-precision was selected over INT8 quantization because the target Tesla T4 GPU natively accelerates FP16 Tensor Core arithmetic, and an INT8 calibration dataset pipeline was not constructed within the allocation time budget.

---

## 7. Benchmark Results

All latency benchmarks represent **ONNX Runtime model inference latency** measured via `InferenceSession.run` on a Tesla T4 GPU (excluding image loading, preprocessing, and NMS postprocessing).

### Benchmark Comparison Table

| Model Format | Model Size | Mean Latency | P95 Latency | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch (FP32)** | 5.46 MB | N/A | N/A | 0.8538 | 0.6548 | 0.7867 | 0.5552 |
| **ONNX Runtime (FP32)** | 10.11 MB | 6.11 ms | 9.69 ms | 0.7232 | 0.7798 | 0.8025 | 0.5224 |
| **ONNX Runtime (FP16)** | 5.11 MB | 4.68 ms | 6.59 ms | 0.8944 | 0.6214 | 0.8015 | 0.5216 |

### FP16 Relative to FP32 Gains & Trade-offs
- **Model Size Reduction**: **49.5% smaller** ($10.11\text{ MB} \rightarrow 5.11\text{ MB}$)
- **Mean Latency Improvement**: **23.4% faster** ($6.11\text{ ms} \rightarrow 4.68\text{ ms}$)
- **P95 Latency Improvement**: **32.0% faster** ($9.69\text{ ms} \rightarrow 6.59\text{ ms}$)
- **mAP@0.5 Change**: $-0.0010$ ($0.8025 \rightarrow 0.8015$)
- **mAP@0.5:0.95 Change**: $-0.0008$ ($0.5224 \rightarrow 0.5216$)

---

## 8. Prediction Parity
Parity was evaluated by comparing inference predictions between PyTorch (`best.pt`), FP32 ONNX (`best_fp32.onnx`), and FP16 ONNX (`best_fp16.onnx`) on a validation image with active object detections:

- **Detection Count**: All 3 model variants detected exactly **2 objects**.
- **Class Predictions**: Identical class assignments across all models (`bottle` and `cup`).
- **Bounding Box Difference**: Maximum bounding box coordinate difference was $\approx 60\text{ px}$.
- **Confidence Score Difference**: Maximum confidence score delta was $\approx 0.066$.

*Conclusion*: Both ONNX models reproduced the same detected objects and classes on the checked validation image, while numerical bounding-box coordinate and confidence score outputs differed.

---

## 9. Part A4 — Failure Analysis

The detailed Part A4 failure analysis, including the original validation images, model predictions, observed failures, hypotheses, and proposed improvements, is provided separately in:

**`A4 - Failure Analysis.pdf`**

---

## 10. Reproduction & Setup

### Environment Setup
```bash
# Clone repository
git clone https://github.com/MukulN7/art-kate-cv-assignment.git
cd art-kate-cv-assignment

# Install dependencies
pip install -r requirements.txt
```

### Dataset Splitting
```bash
python scripts/split_dataset.py
```

### Training
```bash
python scripts/train.py --device 0
```

### ONNX Export
```bash
python scripts/export_onnx.py
```

### ONNX Benchmarking & Evaluation
```bash
python scripts/benchmark_onnx.py
```

---

## 11. Model Artifacts & Checksums

The trained weight files and exported ONNX models are stored under `models/`:

| Artifact Path | Format | Size | SHA256 Checksum |
| :--- | :--- | :---: | :--- |
| [`models/best.pt`](file:///d:/Career/Artikate%20Studio/models/best.pt) | PyTorch Checkpoint | 5.46 MB | `6863DD169DAADF425F7C603DA4E62541CC43A245DECF7D61253341F5586A33B8` |
| [`models/best_fp32.onnx`](file:///d:/Career/Artikate%20Studio/models/best_fp32.onnx) | ONNX FP32 | 10.11 MB | `607FDB58DEF35D407E3D9D1995FCE347E3B063235F9609F811F531B0EB796746` |
| [`models/best_fp16.onnx`](file:///d:/Career/Artikate%20Studio/models/best_fp16.onnx) | ONNX FP16 | 5.11 MB | `835691BAD0D25C661736E93FC3E5C3FE318CECAC94E4EAE497197CAB97F56248` |

---

## 12. Assumptions
1. **Dataset Scope**: The dataset is small (74 images) and self-captured for proof-of-concept evaluation.
2. **Validation Specificity**: Validation metrics reflect performance on the 15-image validation split.
3. **Latency Isolation**: Benchmark timings measure model inference latency on GPU, excluding end-to-end image I/O, resizing, and NMS postprocessing.
4. **Precision Format**: FP16 was chosen as the primary reduced-precision candidate due to hardware acceleration on Tesla T4.

---

## 13. Known Gaps / Limitations
- **Small Dataset Size**: 74 total images limits statistical variance and domain coverage.
- **Physical Object Diversity**: Captured objects represent a limited set of physical bottles and cups.
- **JPEG Header Restoration**: Source JPEG headers required automated software restoration during Ultralytics ingestion.
- **Lack of INT8 Calibration**: INT8 quantization was not implemented due to calibration set complexity and time constraints.
- **Sample-Based Parity**: Parity checks verified representative detection agreement rather than exhaustive floating-point bitwise equality across the complete validation set.

---

## 14. Generalization & Confidence
- **Current Metric Baseline**: The model achieves solid baseline validation performance ($\text{mAP@0.5} = 0.7867$).
- **Generalization Assessment**: Because the dataset was collected under controlled indoor settings, confidence in out-of-domain generalization (e.g., outdoor scenes, extreme lighting, novel container shapes) remains **limited**.
- **Path to Production**: Deploying in unconstrained real-world environments would require expanding dataset diversity across lighting conditions, backgrounds, and object variants.

---

## 15. Assignment Status

- **Part A (Computer Vision & ML Pipeline):** Complete.
- **Parts B, C, D:** Responses will be provided in `ANSWERS.md`.
