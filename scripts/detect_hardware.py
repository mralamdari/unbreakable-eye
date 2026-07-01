#!/usr/bin/env python3
"""
scripts/detect_hardware.py

Detect available hardware and print which requirements file to install.
Run during Docker build or deployment setup — NOT during inference.

Usage:
    python scripts/detect_hardware.py
    python scripts/detect_hardware.py --install   # actually installs
"""

import platform
import subprocess
import sys
import shutil


def check_nvidia() -> bool:
    """True if nvidia-smi is present and reports a GPU."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def check_apple_silicon() -> bool:
    """True if running on Apple Silicon (M1/M2/M3/M4)."""
    return (
        platform.system() == "Darwin" and
        platform.machine() == "arm64"
    )


def check_intel_igpu() -> bool:
    """True if an Intel iGPU/VPU is available (Linux only)."""
    try:
        result = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=5
        )
        output = result.stdout.lower()
        return "intel" in output and ("vga" in output or "display" in output)
    except Exception:
        return False


def detect() -> dict:
    system   = platform.system()      # Darwin, Linux, Windows
    machine  = platform.machine()     # x86_64, arm64, aarch64
    nvidia   = check_nvidia()
    apple_si = check_apple_silicon()
    intel_gpu = check_intel_igpu() if system == "Linux" else False

    return {
        "system":    system,
        "machine":   machine,
        "nvidia":    nvidia,
        "apple_si":  apple_si,
        "intel_gpu": intel_gpu,
    }


def recommend(hw: dict) -> tuple[str, str, str]:
    """
    Returns (onnxruntime_package, extra_requirements_file, explanation)
    """

    if hw["nvidia"]:
        return (
            "onnxruntime-gpu",
            "requirements-gpu.txt",
            "NVIDIA GPU detected — using CUDA-accelerated ONNX Runtime"
        )

    if hw["apple_si"]:
        # CoreML execution provider ships with onnxruntime on macOS arm64
        # No separate package needed — onnxruntime auto-selects CoreML
        return (
            "onnxruntime",
            None,
            "Apple Silicon detected — onnxruntime will use CoreML automatically"
        )

    if hw["intel_gpu"]:
        return (
            "onnxruntime",
            "requirements-optional.txt (openvino)",
            "Intel GPU detected — consider installing openvino for acceleration"
        )

    return (
        "onnxruntime",
        None,
        "CPU-only system — using standard ONNX Runtime"
    )


def main():
    install = "--install" in sys.argv

    hw = detect()
    pkg, extra_req, explanation = recommend(hw)

    print("=== Hardware Detection ===")
    print(f"  OS:           {hw['system']} {hw['machine']}")
    print(f"  NVIDIA GPU:   {'YES' if hw['nvidia'] else 'no'}")
    print(f"  Apple Silicon:{'YES' if hw['apple_si'] else 'no'}")
    print(f"  Intel GPU:    {'YES' if hw['intel_gpu'] else 'no'}")
    print()
    print(f"=== Recommendation ===")
    print(f"  {explanation}")
    print(f"  ONNX Runtime package: {pkg}")
    if extra_req:
        print(f"  Extra requirements:   {extra_req}")
    print()

    if install:
        print("Installing...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "-r", "requirements.txt"],
            check=True
        )
        if extra_req and "requirements-" in extra_req:
            req_file = extra_req.split()[0]
            subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "-r", req_file],
                check=True
            )
        # Replace CPU onnxruntime with GPU version if needed
        if pkg != "onnxruntime":
            subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 f"{pkg}==1.23.2", "--force-reinstall"],
                check=True
            )
        print("Done.")
    else:
        print("Run with --install to apply these recommendations.")


if __name__ == "__main__":
    main()


