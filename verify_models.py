"""Script to download and verify real AI models for translate-en-es.

This script initializes the core engines with real models (not mocks)
and forces them to load, triggering downloads if they don't exist.
"""

import asyncio
from translator.core.config import AppConfig
from translator.infrastructure.gpu.gpu_model_manager import GPUModelManager
from translator.infrastructure.vad.silero_vad import SileroVADEngine
from translator.infrastructure.asr.whisper_asr import WhisperASREngine
from translator.infrastructure.mt.marian_mt import MarianMTEngine

async def verify() -> None:
    print("Initializing config...")
    config = AppConfig()
    
    print("Initializing GPU Model Manager...")
    manager = GPUModelManager(max_vram_mb=config.gpu.max_vram_mb)
    print(f"Detected Device: {manager.detect_device()}")
    
    print("\n--- Verifying Silero VAD ---")
    vad = SileroVADEngine(config.vad, manager)
    await vad._load_model()
    print("Silero VAD loaded successfully.")
    
    print("\n--- Verifying Faster-Whisper ---")
    asr = WhisperASREngine(config.asr, manager)
    await asr.load_model()
    print("Whisper loaded successfully.")
    
    print("\n--- Verifying MarianMT ---")
    # For speed, use the lightweight model we configured
    mt = MarianMTEngine(config.mt, manager)
    await mt.load_model()
    print("MarianMT loaded successfully.")
    
    print("\n--- VRAM Status ---")
    status = manager.get_vram_status()
    print(f"Total VRAM: {status.total_mb} MB")
    print(f"Used VRAM: {status.used_mb} MB")
    print(f"Free VRAM: {status.free_mb} MB")
    print(f"Loaded Models: {[m.name for m in status.models]}")

if __name__ == "__main__":
    asyncio.run(verify())
