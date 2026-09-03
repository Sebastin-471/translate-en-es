"""Script to download and verify real AI models for translate-en-es.

This script initializes the core engines with real models (not mocks)
and forces them to load, triggering downloads if they don't exist.

Also verifies prepared models against model_manifest.json if available.
"""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

from translator.core.config import AppConfig
from translator.infrastructure.gpu.gpu_model_manager import GPUModelManager
from translator.infrastructure.vad.silero_vad import SileroVADEngine
from translator.infrastructure.asr.whisper_asr import WhisperASREngine
from translator.infrastructure.mt.marian_mt import MarianMTEngine


def compute_dir_sha256(dirpath: Path) -> str:
    """Compute SHA256 hash of all files in a directory (sorted)."""
    sha256 = hashlib.sha256()
    for file_path in sorted(dirpath.rglob("*")):
        if file_path.is_file():
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            sha256.update(str(file_path.relative_to(dirpath)).encode())
    return sha256.hexdigest()


def verify_manifest(models_dir: Path) -> bool:
    """Verify models against model_manifest.json."""
    manifest_path = models_dir / "model_manifest.json"
    if not manifest_path.exists():
        print(f"No manifest found at {manifest_path}, skipping manifest verification")
        return True

    print(f"\n--- Verifying against manifest: {manifest_path} ---")
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_passed = True
    for model_data in manifest["models"]:
        model_name = model_data["name"]
        expected_hash = model_data["sha256"]
        model_path = models_dir / model_name

        if not model_path.exists():
            print(f"  FAIL: {model_name} - path not found: {model_path}")
            all_passed = False
            continue

        if model_path.is_dir():
            actual_hash = compute_dir_sha256(model_path)
        else:
            with open(model_path, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()

        if actual_hash == expected_hash:
            print(f"  OK: {model_name} (hash matches)")
        else:
            print(f"  FAIL: {model_name} - hash mismatch!")
            print(f"    Expected: {expected_hash}")
            print(f"    Actual:   {actual_hash}")
            all_passed = False

    return all_passed


async def verify() -> int:
    print("Initializing config...")
    config = AppConfig()

    print("Initializing GPU Model Manager...")
    manager = GPUModelManager(max_vram_mb=config.gpu.max_vram_mb)
    print(f"Detected Device: {manager.detect_device()}")

    # Check for prepared models
    models_dir = Path("models")
    manifest_verified = True
    if models_dir.exists():
        manifest_verified = verify_manifest(models_dir)

    print("\n--- Verifying Silero VAD ---")
    vad = SileroVADEngine(config.vad, manager)
    await vad._load_model()
    print("Silero VAD loaded successfully.")

    print("\n--- Verifying Faster-Whisper ---")
    asr = WhisperASREngine(config.asr, manager)
    await asr.load_model()
    print("Whisper loaded successfully.")

    print("\n--- Verifying MarianMT ---")
    mt = MarianMTEngine(config.mt, manager)
    await mt.load_model()
    print("MarianMT loaded successfully.")

    print("\n--- VRAM Status ---")
    status = manager.get_vram_status()
    print(f"Total VRAM: {status.total_mb:.0f} MB")
    print(f"Used VRAM: {status.used_mb:.0f} MB")
    print(f"Free VRAM: {status.free_mb:.0f} MB")
    print(f"Loaded Models: {[m.name for m in status.models]}")

    if not manifest_verified:
        print("\nWARNING: Model manifest verification failed!")
        return 1

    print("\nAll models verified successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(verify()))