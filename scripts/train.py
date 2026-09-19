#!/usr/bin/env python3
"""
Minimal, reproducible training script for YOLO11n on the Bottle & Cup dataset.
Configured for execution on Google Colab or local GPU environments.
"""

import argparse
import sys
from pathlib import Path

# Try importing torch to detect CUDA availability gracefully
try:
    import torch
    DEFAULT_DEVICE = "0" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEFAULT_DEVICE = "cpu"

# Determine repository root directory using relative path logic
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = REPO_ROOT / "data" / "data.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO11n model on Bottle & Cup dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLO model checkpoint (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_YAML),
        help="Path to dataset configuration YAML (default: data/data.yaml)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training and validation (default: 640)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        help="Optimizer choice (default: auto)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs (default: 15)"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable Automatic Mixed Precision training (default: True)"
    )
    parser.add_argument(
        "--no-amp",
        action="store_false",
        dest="amp",
        help="Disable Automatic Mixed Precision training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device to run training on, e.g., '0' or 'cpu' (default: {DEFAULT_DEVICE})"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(REPO_ROOT / "runs" / "detect"),
        help="Project output directory for training runs"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo11n_bottle_cup",
        help="Name of the training run directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Print actual training configuration
    print("=" * 60)
    print("YOLO11n Training Configuration")
    print("=" * 60)
    print(f"  Model Checkpoint : {args.model}")
    print(f"  Dataset YAML     : {args.data}")
    print(f"  Epochs           : {args.epochs}")
    print(f"  Image Size       : {args.imgsz}")
    print(f"  Batch Size       : {args.batch}")
    print(f"  Optimizer        : {args.optimizer}")
    print(f"  Seed             : {args.seed}")
    print(f"  Patience         : {args.patience}")
    print(f"  AMP Enabled      : {args.amp}")
    print(f"  Device           : {args.device}")
    print(f"  Project Dir      : {args.project}")
    print(f"  Run Name         : {args.name}")
    print("=" * 60)

    # Validate data.yaml existence
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset configuration file not found at {data_path.resolve()}")
        sys.exit(1)

    # Import ultralytics YOLO model
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: 'ultralytics' package is not installed.")
        print("Please install via: pip install ultralytics")
        sys.exit(1)

    # Load pretrained model
    print(f"\nLoading model: {args.model}...")
    model = YOLO(args.model)

    # Start training using Ultralytics API
    print("\nStarting training...\n")
    results = model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        seed=args.seed,
        patience=args.patience,
        amp=args.amp,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("Training Completed Successfully!")
    print(f"Results and weights saved to: {Path(args.project) / args.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
