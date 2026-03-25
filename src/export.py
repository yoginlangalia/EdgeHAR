"""
Model Export Script for EdgeHAR.

Exports the trained CNN-LSTM model to:
    - ONNX format (for cross-platform inference, e.g., Streamlit dashboard)
    - TorchScript format (for C++/mobile deployment)

Performs verification that exported models produce identical outputs
and benchmarks inference speed for all formats.

Usage:
    python src/export.py
"""

import time
from pathlib import Path

import numpy as np
import torch

from model import CNNLSTM

# ─── Configuration ───────────────────────────────────────────────────────────
CONFIG = {
    "model_path": Path(__file__).resolve().parent.parent / "models" / "best_model.pth",
    "onnx_path": Path(__file__).resolve().parent.parent / "models" / "har_model.onnx",
    "torchscript_path": Path(__file__).resolve().parent.parent / "models" / "har_model_scripted.pt",
    "seed": 42,
    "opset_version": 12,
    "num_benchmark_runs": 100,
    "input_shape": (1, 6, 128),
}


def load_trained_model(device: torch.device) -> tuple:
    """Load the trained model from checkpoint.

    Args:
        device: Device to load the model on.

    Returns:
        Tuple of (model, checkpoint).

    Raises:
        FileNotFoundError: If the model checkpoint doesn't exist.
    """
    model_path = CONFIG["model_path"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ Model checkpoint not found at: {model_path}\n"
            f"   Please run training first: python src/train.py"
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})

    model = CNNLSTM(
        num_channels=model_config.get("num_channels", 6),
        num_classes=model_config.get("num_classes", 6),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"📥 Loaded model from: {model_path}")
    print(f"   Epoch: {checkpoint.get('epoch', '?')} | "
          f"Val Acc: {checkpoint.get('val_acc', 0):.2%}")

    return model, checkpoint


def export_to_onnx(model: torch.nn.Module, device: torch.device) -> Path:
    """Export model to ONNX format.

    Args:
        model: Trained PyTorch model.
        device: Device the model is on.

    Returns:
        Path to the saved ONNX model.
    """
    onnx_path = CONFIG["onnx_path"]
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*CONFIG["input_shape"], device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=CONFIG["opset_version"],
        do_constant_folding=True,
        input_names=["sensor_input"],
        output_names=["activity_output"],
        dynamic_axes={
            "sensor_input": {0: "batch_size"},
            "activity_output": {0: "batch_size"},
        },
    )

    print(f"\n✅ ONNX model exported to: {onnx_path}")
    return onnx_path


def export_to_torchscript(model: torch.nn.Module, device: torch.device) -> Path:
    """Export model to TorchScript format.

    Args:
        model: Trained PyTorch model.
        device: Device the model is on.

    Returns:
        Path to the saved TorchScript model.
    """
    ts_path = CONFIG["torchscript_path"]
    ts_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(*CONFIG["input_shape"], device=device)
    scripted_model = torch.jit.trace(model, dummy_input)
    scripted_model.save(str(ts_path))

    print(f"✅ TorchScript model exported to: {ts_path}")
    return ts_path


def verify_exports(model: torch.nn.Module, device: torch.device) -> None:
    """Verify that exported models produce identical outputs.

    Compares ONNX and TorchScript outputs against the original PyTorch
    model output. Asserts max absolute difference < 1e-4.

    Args:
        model: Original PyTorch model.
        device: Device to run on.

    Raises:
        AssertionError: If output differences exceed threshold.
    """
    print("\n🔍 Verifying exported models...")

    torch.manual_seed(CONFIG["seed"])
    test_input = torch.randn(*CONFIG["input_shape"], device=device)

    # PyTorch reference output
    with torch.no_grad():
        pytorch_output = model(test_input).cpu().numpy()

    # ─── Verify ONNX ────────────────────────────────────────────────────
    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(str(CONFIG["onnx_path"]))
        onnx.checker.check_model(onnx_model)
        print("   ✅ ONNX model passed validation check")

        session = ort.InferenceSession(str(CONFIG["onnx_path"]))
        onnx_output = session.run(
            None, {"sensor_input": test_input.cpu().numpy()}
        )[0]

        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        assert max_diff < 1e-4, f"ONNX output mismatch: max diff = {max_diff}"
        print(f"   ✅ ONNX output matches PyTorch (max diff: {max_diff:.2e})")

    except ImportError:
        print("   ⚠️  onnx/onnxruntime not installed, skipping ONNX verification")

    # ─── Verify TorchScript ──────────────────────────────────────────────
    scripted_model = torch.jit.load(str(CONFIG["torchscript_path"]), map_location=device)
    with torch.no_grad():
        ts_output = scripted_model(test_input).cpu().numpy()

    max_diff = np.max(np.abs(pytorch_output - ts_output))
    assert max_diff < 1e-4, f"TorchScript output mismatch: max diff = {max_diff}"
    print(f"   ✅ TorchScript output matches PyTorch (max diff: {max_diff:.2e})")


def print_model_sizes() -> None:
    """Print file sizes of all exported model formats."""
    print("\n📦 Model Sizes:")

    paths = {
        "PyTorch (.pth)": CONFIG["model_path"],
        "ONNX (.onnx)": CONFIG["onnx_path"],
        "TorchScript (.pt)": CONFIG["torchscript_path"],
    }

    for name, path in paths.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"   {name:25s} : {size_kb:>8.1f} KB")
        else:
            print(f"   {name:25s} : not found")


def benchmark_inference(model: torch.nn.Module, device: torch.device) -> None:
    """Benchmark inference time for all model formats.

    Runs inference N times and reports average time.

    Args:
        model: Original PyTorch model.
        device: Device to run on.
    """
    n_runs = CONFIG["num_benchmark_runs"]
    test_input = torch.randn(*CONFIG["input_shape"], device=device)

    print(f"\n⏱️  Inference Benchmark ({n_runs} runs):")

    # PyTorch
    model.eval()
    with torch.no_grad():
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(test_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    avg_pytorch = np.mean(times) * 1000
    print(f"   PyTorch       : {avg_pytorch:.3f} ms/sample")

    # ONNX Runtime
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(CONFIG["onnx_path"]))
        np_input = test_input.cpu().numpy()
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = session.run(None, {"sensor_input": np_input})
            times.append(time.perf_counter() - start)
        avg_onnx = np.mean(times) * 1000
        print(f"   ONNX Runtime  : {avg_onnx:.3f} ms/sample")
    except ImportError:
        print("   ONNX Runtime  : skipped (not installed)")

    # TorchScript
    scripted_model = torch.jit.load(str(CONFIG["torchscript_path"]), map_location=device)
    with torch.no_grad():
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = scripted_model(test_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    avg_ts = np.mean(times) * 1000
    print(f"   TorchScript   : {avg_ts:.3f} ms/sample")


def main() -> None:
    """Main export function."""
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}\n")

    # Load trained model
    model, _ = load_trained_model(device)

    # Export
    export_to_onnx(model, device)
    export_to_torchscript(model, device)

    # Verify
    verify_exports(model, device)

    # Stats
    print_model_sizes()
    benchmark_inference(model, device)

    print("\n✅ All exports complete and verified!")


if __name__ == "__main__":
    main()
