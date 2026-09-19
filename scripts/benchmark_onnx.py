#!/usr/bin/env python3
"""
Benchmark script for ONNX Runtime FP32 and FP16 models vs PyTorch YOLO11n.
Measures model size, mean/P95 latency, precision, recall, mAP@0.5, mAP@0.5:0.95,
and compares predictions (PyTorch vs FP32 ONNX and PyTorch vs FP16 ONNX).
Part A3 - Minimal Reproducible Benchmark Suite.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Repository root and default paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PT = REPO_ROOT / "runs" / "detect" / "yolo11n_bottle_cup" / "weights" / "best.pt"
DEFAULT_FP32_ONNX = REPO_ROOT / "runs" / "detect" / "yolo11n_bottle_cup" / "weights" / "best_fp32.onnx"
DEFAULT_FP16_ONNX = REPO_ROOT / "runs" / "detect" / "yolo11n_bottle_cup" / "weights" / "best_fp16.onnx"
DEFAULT_VAL_DATA = REPO_ROOT / "data" / "data.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX Runtime FP32 and FP16 models against PyTorch YOLO11n."
    )
    parser.add_argument(
        "--pt-path",
        type=str,
        default=str(DEFAULT_PT),
        help="Path to PyTorch best.pt model"
    )
    parser.add_argument(
        "--fp32-path",
        type=str,
        default=str(DEFAULT_FP32_ONNX),
        help="Path to FP32 ONNX model"
    )
    parser.add_argument(
        "--fp16-path",
        type=str,
        default=str(DEFAULT_FP16_ONNX),
        help="Path to FP16 ONNX model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_VAL_DATA),
        help="Path to data.yaml dataset config"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference (default: 640)"
    )
    return parser.parse_args()


def get_model_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def get_onnx_providers():
    """
    Selects CUDAExecutionProvider if available, otherwise reports CPU fallback.
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            print("  [HARDWARE] CUDAExecutionProvider is available and selected.")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            print("  [NOTICE] CUDAExecutionProvider is unavailable. Falling back to CPUExecutionProvider.")
            return ["CPUExecutionProvider"]
    except ImportError:
        print("Error: 'onnxruntime' is not installed.")
        sys.exit(1)


def benchmark_onnx_latency(onnx_path: Path, is_fp16: bool, warmup: int, runs: int, imgsz: int, providers: list):
    """
    Measures mean and P95 latency for an ONNX model using ONNX Runtime.
    Reports the actual execution provider used by the active session.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    actual_providers = session.get_providers()

    input_name = session.get_inputs()[0].name
    dtype = np.float16 if is_fp16 else np.float32
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(dtype)

    # Warmup iterations
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})

    # Benchmark iterations
    latencies = []
    for _ in range(runs):
        start_time = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies.append(elapsed_ms)

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    return mean_latency, p95_latency, actual_providers


def evaluate_model_metrics(model_path: Path, data_path: Path):
    """
    Evaluates detection metrics (Precision, Recall, mAP@0.5, mAP@0.5:0.95) on the validation set.
    """
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        metrics = model.val(data=str(data_path), split="val", verbose=False)
        return {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        }
    except Exception as e:
        print(f"Warning: Validation evaluation failed for {model_path.name}: {e}")
        return {"precision": 0.0, "recall": 0.0, "mAP50": 0.0, "mAP50-95": 0.0}


def compare_predictions(pt_path: Path, onnx_path: Path, val_image_path: Path):
    """
    Compares predictions between PyTorch model and ONNX model on a sample validation image.
    Explicitly reports detection count mismatches.
    """
    try:
        from ultralytics import YOLO
        pt_model = YOLO(str(pt_path))
        onnx_model = YOLO(str(onnx_path))

        res_pt = pt_model.predict(str(val_image_path), verbose=False)[0]
        res_onnx = onnx_model.predict(str(val_image_path), verbose=False)[0]

        pt_boxes = res_pt.boxes.xyxy.cpu().numpy()
        onnx_boxes = res_onnx.boxes.xyxy.cpu().numpy()

        pt_confs = res_pt.boxes.conf.cpu().numpy()
        onnx_confs = res_onnx.boxes.conf.cpu().numpy()

        pt_cls = res_pt.boxes.cls.cpu().numpy()
        onnx_cls = res_onnx.boxes.cls.cpu().numpy()

        num_pt = len(pt_boxes)
        num_onnx = len(onnx_boxes)

        if num_pt != num_onnx:
            return {
                "num_pt": num_pt,
                "num_onnx": num_onnx,
                "mismatch": True,
                "note": f"Detection count mismatch: PyTorch found {num_pt}, ONNX found {num_onnx}",
                "max_box_diff": None,
                "max_conf_diff": None,
                "class_match": False
            }

        if num_pt == 0:
            return {
                "num_pt": 0,
                "num_onnx": 0,
                "mismatch": False,
                "note": "No objects detected by either model",
                "max_box_diff": 0.0,
                "max_conf_diff": 0.0,
                "class_match": True
            }

        max_box_diff = float(np.max(np.abs(pt_boxes - onnx_boxes)))
        max_conf_diff = float(np.max(np.abs(pt_confs - onnx_confs)))
        class_match = bool(np.array_equal(pt_cls, onnx_cls))

        return {
            "num_pt": num_pt,
            "num_onnx": num_onnx,
            "mismatch": False,
            "note": "Matching detection counts",
            "max_box_diff": max_box_diff,
            "max_conf_diff": max_conf_diff,
            "class_match": class_match
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    args = parse_args()
    print("=" * 75)
    print("ONNX Runtime & PyTorch Model Benchmark Suite")
    print("=" * 75)
    providers = get_onnx_providers()
    print(f"  PyTorch Model (.pt)   : {args.pt_path}")
    print(f"  FP32 ONNX Model       : {args.fp32_path}")
    print(f"  FP16 ONNX Model       : {args.fp16_path}")
    print(f"  Dataset Config        : {args.data}")
    print(f"  Warmup Iterations     : {args.warmup}")
    print(f"  Benchmark Iterations  : {args.runs}")
    print("=" * 75)

    pt_path = Path(args.pt_path)
    fp32_path = Path(args.fp32_path)
    fp16_path = Path(args.fp16_path)
    data_path = Path(args.data)

    models_to_test = [
        ("PyTorch (FP32)", pt_path, False),
        ("ONNX Runtime (FP32)", fp32_path, False),
        ("ONNX Runtime (FP16)", fp16_path, True),
    ]

    results_table = []

    for name, path, is_fp16 in models_to_test:
        print(f"\n--- Benchmarking: {name} ---")
        if not path.exists():
            print(f"  [SKIPPED] Model file not found at: {path}")
            continue

        size_mb = get_model_size_mb(path)
        print(f"  File Size          : {size_mb:.2f} MB")

        # Benchmark Latency via ONNX Runtime
        if path.suffix == ".onnx":
            try:
                mean_lat, p95_lat, active_providers = benchmark_onnx_latency(
                    path, is_fp16, args.warmup, args.runs, args.imgsz, providers
                )
                print(f"  Active Provider(s) : {active_providers}")
                print(f"  Mean Latency       : {mean_lat:.2f} ms")
                print(f"  P95 Latency        : {p95_lat:.2f} ms")
            except Exception as e:
                print(f"  [FAIL] Latency benchmark failed: {e}")
                mean_lat, p95_lat = None, None
        else:
            mean_lat, p95_lat = None, None

        # Evaluate Validation Metrics
        print("  Evaluating validation metrics (Precision, Recall, mAP)...")
        metrics = evaluate_model_metrics(path, data_path)
        print(f"  Precision          : {metrics['precision']:.4f}")
        print(f"  Recall             : {metrics['recall']:.4f}")
        print(f"  mAP@0.5            : {metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95       : {metrics['mAP50-95']:.4f}")

        results_table.append({
            "name": name,
            "size_mb": size_mb,
            "mean_lat": mean_lat,
            "p95_lat": p95_lat,
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "mAP50": metrics['mAP50'],
            "mAP50-95": metrics['mAP50-95'],
        })

    # Summary Table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Model':<22} | {'Size (MB)':<9} | {'Mean (ms)':<9} | {'P95 (ms)':<9} | {'Precision':<9} | {'Recall':<9} | {'mAP50':<9} | {'mAP50-95':<9}"
    print(header)
    print("-" * len(header))
    for r in results_table:
        mean_str = f"{r['mean_lat']:.2f}" if r['mean_lat'] is not None else "N/A"
        p95_str = f"{r['p95_lat']:.2f}" if r['p95_lat'] is not None else "N/A"
        print(f"{r['name']:<22} | {r['size_mb']:<9.2f} | {mean_str:<9} | {p95_str:<9} | {r['precision']:<9.4f} | {r['recall']:<9.4f} | {r['mAP50']:<9.4f} | {r['mAP50-95']:<9.4f}")
    print("=" * 80)

    # Compare Predictions on sample validation image (PyTorch vs FP32 ONNX and PyTorch vs FP16 ONNX)
    val_img_dir = REPO_ROOT / "data" / "val" / "images"
    sample_images = sorted(list(val_img_dir.glob("*.jpg")) + list(val_img_dir.glob("*.jpeg")))
    if sample_images and pt_path.exists():
        from ultralytics import YOLO
        pt_finder = YOLO(str(pt_path))
        sample_img = None
        sample_det_count = 0

        for img_p in sample_images:
            res = pt_finder.predict(str(img_p), verbose=False)[0]
            if len(res.boxes) > 0:
                sample_img = img_p
                sample_det_count = len(res.boxes)
                break

        if sample_img is None:
            sample_img = sample_images[0]

        print(f"\nSelected image for parity check: {sample_img.name} (PyTorch detections: {sample_det_count})")
        print(f"--- Prediction Parity Comparison on Sample Image: {sample_img.name} ---")

        for onnx_label, onnx_p in [("PyTorch vs FP32 ONNX", fp32_path), ("PyTorch vs FP16 ONNX", fp16_path)]:
            if not onnx_p.exists():
                print(f"  [{onnx_label}] Model file not found at: {onnx_p}")
                continue

            print(f"\n  Comparison: {onnx_label}")
            comp = compare_predictions(pt_path, onnx_p, sample_img)
            if "error" not in comp:
                print(f"    Status                   : {comp['note']}")
                print(f"    PyTorch Detections       : {comp['num_pt']}")
                print(f"    ONNX Detections          : {comp['num_onnx']}")
                if not comp['mismatch']:
                    print(f"    Max Box Coordinate Diff  : {comp['max_box_diff']:.6f} px")
                    print(f"    Max Confidence Diff      : {comp['max_conf_diff']:.6f}")
                    print(f"    Class Predictions Match  : {comp['class_match']}")
            else:
                print(f"    Comparison error: {comp['error']}")


if __name__ == "__main__":
    main()
