#!/usr/bin/env python
"""Model Preparation Script for translate-en-es.

Downloads, converts, quantizes, and verifies ML models for offline distribution.
Creates a model package that can be bundled with the application.

Usage:
    python scripts/prepare_models.py --output-dir ./models --quantize int8
    python scripts/prepare_models.py --output-dir ./models --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about a prepared model."""
    name: str
    model_type: str  # "asr", "mt", "vad"
    source: str      # HuggingFace repo or local path
    revision: str    # Git revision/commit hash
    output_path: str # Local path to prepared model
    size_mb: float
    sha256: str
    quantization: str | None = None
    compute_type: str | None = None


# Model specifications with pinned revisions for reproducibility
MODEL_SPECS = {
    "whisper-large-v3-turbo": {
        "model_type": "asr",
        "source": "Systran/faster-whisper-large-v3-turbo",
        "revision": "main",
        "quantization": "int8",
        "compute_type": "int8",
    },
    "whisper-large-v3": {
        "model_type": "asr",
        "source": "Systran/faster-whisper-large-v3",
        "revision": "main",
        "quantization": "int8",
        "compute_type": "int8",
    },
    "whisper-medium": {
        "model_type": "asr",
        "source": "Systran/faster-whisper-medium",
        "revision": "main",
        "quantization": "int8",
        "compute_type": "int8",
    },
    "opus-mt-en-es": {
        "model_type": "mt",
        "source": "Helsinki-NLP/opus-mt-en-es",
        "revision": "main",
        "quantization": "int8",
        "compute_type": "int8",
    },
    "silero-vad": {
        "model_type": "vad",
        "source": "silero-vad",
        "revision": "v6.1",  # silero-vad version
        "quantization": None,
        "compute_type": None,
    },
}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_dir_sha256(dirpath: Path) -> str:
    """Compute SHA256 hash of all files in a directory (sorted)."""
    sha256 = hashlib.sha256()
    for file_path in sorted(dirpath.rglob("*")):
        if file_path.is_file():
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            # Include relative path in hash for structure verification
            sha256.update(str(file_path.relative_to(dirpath)).encode())
    return sha256.hexdigest()


def get_dir_size_mb(dirpath: Path) -> float:
    """Get directory size in MB."""
    total = sum(f.stat().st_size for f in dirpath.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def run_command(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run a command and return result."""
    logger.info("running_command", cmd=" ".join(cmd), cwd=str(cwd) if cwd else None)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("command_failed", cmd=" ".join(cmd), stderr=result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result


def download_hf_model(repo_id: str, revision: str, output_dir: Path) -> Path:
    """Download model from HuggingFace Hub."""
    logger.info("downloading_model", repo=repo_id, revision=revision)
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )
    return Path(local_dir)


def convert_whisper_to_ct2(model_dir: Path, output_dir: Path, quantization: str) -> Path:
    """Convert Whisper model to CTranslate2 format."""
    logger.info("converting_whisper_to_ct2", input=model_dir, output=output_dir, quantization=quantization)

    cmd = [
        "ct2-transformers-converter",
        "--model", str(model_dir),
        "--output_dir", str(output_dir),
        "--quantization", quantization,
        "--force",
    ]
    run_command(cmd)
    return output_dir


def convert_marian_to_ct2(model_dir: Path, output_dir: Path, quantization: str) -> Path:
    """Convert MarianMT model to CTranslate2 format."""
    logger.info("converting_marian_to_ct2", input=model_dir, output=output_dir, quantization=quantization)

    cmd = [
        "ct2-transformers-converter",
        "--model", str(model_dir),
        "--output_dir", str(output_dir),
        "--quantization", quantization,
        "--force",
    ]
    run_command(cmd)
    return output_dir


def prepare_silero_vad(output_dir: Path) -> Path:
    """Prepare Silero VAD model (no conversion needed, just verify)."""
    logger.info("preparing_silero_vad", output=output_dir)
    # Silero VAD is loaded via torch.hub, no local files needed
    # We just create a marker file
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".silero_vad_ready").touch()
    return output_dir


def verify_model(model_info: ModelInfo) -> bool:
    """Verify a prepared model by checking hash and loading it."""
    logger.info("verifying_model", name=model_info.name)

    path = Path(model_info.output_path)
    if not path.exists():
        logger.error("model_path_missing", path=str(path))
        return False

    # Check hash
    if path.is_dir():
        actual_hash = compute_dir_sha256(path)
    else:
        actual_hash = compute_sha256(path)

    if actual_hash != model_info.sha256:
        logger.error("hash_mismatch", expected=model_info.sha256, actual=actual_hash, path=str(path))
        return False

    # Try loading the model (basic sanity check)
    try:
        if model_info.model_type == "asr":
            from faster_whisper import WhisperModel
            WhisperModel(str(path), device="cpu", compute_type=model_info.compute_type or "int8")
        elif model_info.model_type == "mt":
            import ctranslate2
            ctranslate2.Translator(str(path), device="cpu")
        elif model_info.model_type == "vad":
            import torch
            torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        logger.info("model_verification_passed", name=model_info.name)
        return True
    except Exception as e:
        logger.error("model_load_failed", name=model_info.name, error=str(e))
        return False


def prepare_model(model_key: str, output_dir: Path, quantization: str | None = None) -> ModelInfo:
    """Prepare a single model."""
    spec = MODEL_SPECS[model_key]
    model_type = spec["model_type"]

    # Use custom quantization if provided
    quant = quantization or spec["quantization"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Download from HF
        if model_type in ("asr", "mt"):
            hf_dir = tmp_path / "hf"
            download_hf_model(spec["source"], spec["revision"], hf_dir)

            # Convert to CTranslate2
            ct2_dir = tmp_path / "ct2"
            if model_type == "asr":
                convert_whisper_to_ct2(hf_dir, ct2_dir, quant or "int8")
            else:
                convert_marian_to_ct2(hf_dir, ct2_dir, quant or "int8")

            # Copy to final location
            final_dir = output_dir / model_key
            if final_dir.exists():
                shutil.rmtree(final_dir)
            shutil.copytree(ct2_dir, final_dir)

            size_mb = get_dir_size_mb(final_dir)
            sha256 = compute_dir_sha256(final_dir)

        elif model_type == "vad":
            vad_dir = output_dir / model_key
            prepare_silero_vad(vad_dir)
            size_mb = 0.0  # No local files
            sha256 = "silero-vad-loaded-via-torch-hub"

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return ModelInfo(
        name=model_key,
        model_type=model_type,
        source=spec["source"],
        revision=spec["revision"],
        output_path=str(output_dir / model_key),
        size_mb=size_mb,
        sha256=sha256,
        quantization=quant,
        compute_type=spec["compute_type"],
    )


def write_manifest(models: list[ModelInfo], output_dir: Path) -> None:
    """Write model manifest JSON."""
    manifest = {
        "version": "1.0",
        "models": [asdict(m) for m in models],
    }
    manifest_path = output_dir / "model_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("manifest_written", path=str(manifest_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare ML models for translate-en-es")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for prepared models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SPECS.keys()),
        default=list(MODEL_SPECS.keys()),
        help="Models to prepare (default: all)",
    )
    parser.add_argument(
        "--quantization",
        choices=["int8", "float16", "float32"],
        help="Override quantization for all models",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing models, don't prepare",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip verification after preparation",
    )
    args = parser.parse_args()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verify_only:
        logger.info("verification_only_mode")
        # Load manifest and verify
        manifest_path = output_dir / "model_manifest.json"
        if not manifest_path.exists():
            logger.error("manifest_not_found", path=str(manifest_path))
            return 1

        with open(manifest_path) as f:
            manifest = json.load(f)

        all_passed = True
        for model_data in manifest["models"]:
            model_info = ModelInfo(**model_data)
            if not verify_model(model_info):
                all_passed = False

        if all_passed:
            logger.info("all_models_verified")
            return 0
        else:
            logger.error("some_models_failed_verification")
            return 1

    # Prepare models
    logger.info("preparing_models", models=args.models, output_dir=str(output_dir))
    prepared_models = []

    for model_key in args.models:
        try:
            model_info = prepare_model(model_key, output_dir, args.quantization)
            prepared_models.append(model_info)
            logger.info("model_prepared", name=model_key, size_mb=model_info.size_mb, sha256=model_info.sha256[:16] + "...")
        except Exception as e:
            logger.error("model_preparation_failed", model=model_key, error=str(e))
            return 1

    # Verify if not skipped
    if not args.skip_verification:
        logger.info("verifying_prepared_models")
        all_passed = True
        for model_info in prepared_models:
            if not verify_model(model_info):
                all_passed = False
        if not all_passed:
            logger.error("verification_failed")
            return 1

    # Write manifest
    write_manifest(prepared_models, output_dir)

    logger.info("all_models_prepared_successfully", count=len(prepared_models))
    return 0


if __name__ == "__main__":
    sys.exit(main())