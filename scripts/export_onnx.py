#!/usr/bin/env python3
"""
Export trained YOLO11n PyTorch model (.pt) to ONNX format (FP32 & FP16).
Part A3 - Minimal Reproducible Export Pipeline.
"""

import argparse
import sys
from pathlib import Path

# Repository root directory
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "runs" / "detect" / "yolo11n_bottle_cup" / "weights" / "best.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export YOLO11n PyTorch model to ONNX (FP32 & FP16)."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to trained PyTorch weights file (.pt)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export image size (default: 640)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory to save exported ONNX models (default: same directory as weights)"
    )
    return parser.parse_args()


def export_models():
    args = parse_args()
    weights_path = Path(args.weights)

    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path.resolve()}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else weights_path.parent

    try:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: 'ultralytics' package is required for model export.")
        sys.exit(1)

    print("=" * 60)
    print("YOLO11n ONNX Export Configuration")
    print("=" * 60)
    print(f"  Weights Path : {weights_path.resolve()}")
    print(f"  Image Size   : {args.imgsz}x{args.imgsz} (Static)")
    print(f"  Opset        : {args.opset}")
    print(f"  Device       : {device}")
    print(f"  Output Dir   : {output_dir.resolve()}")
    print("=" * 60)

    # 1. Export FP32 ONNX Model
    print("\n[1/2] Exporting FP32 ONNX model...")
    try:
        model_fp32 = YOLO(str(weights_path))
        exported_fp32_path = model_fp32.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            half=False,
            dynamic=False,
            device=device
        )
        fp32_onnx = Path(exported_fp32_path)
        target_fp32 = output_dir / "best_fp32.onnx"
        if fp32_onnx != target_fp32:
            import shutil
            shutil.move(str(fp32_onnx), str(target_fp32))
        print(f"  [SUCCESS] FP32 ONNX model saved to: {target_fp32.resolve()}")
    except Exception as e:
        print(f"Error: FP32 ONNX export failed: {e}")
        sys.exit(1)

    # 2. Export FP16 ONNX Model
    print("\n[2/2] Exporting FP16 ONNX model...")
    try:
        model_fp16 = YOLO(str(weights_path))
        exported_fp16_path = model_fp16.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            half=True,
            dynamic=False,
            device=device
        )
        fp16_onnx = Path(exported_fp16_path)
        target_fp16 = output_dir / "best_fp16.onnx"
        if fp16_onnx != target_fp16:
            import shutil
            shutil.move(str(fp16_onnx), str(target_fp16))
        print(f"  [SUCCESS] FP16 ONNX model saved to: {target_fp16.resolve()}")
    except Exception as e:
        print(f"  [UNAVAILABLE] FP16 ONNX export is unavailable or failed on this device: {e}")

    print("\n" + "=" * 60)
    print("ONNX Export Process Completed!")
    print("=" * 60)


if __name__ == "__main__":
    export_models()
