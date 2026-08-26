"""Quick diagnostic tool: captures 5 seconds of system audio and reports stats.

Usage:
    python tools/diagnose_audio.py

This script bypasses the full pipeline and directly tests:
  1. WASAPI loopback device detection
  2. Audio capture and RMS amplitude
  3. Silero VAD speech detection

If the RMS is near zero (-100 dB), the program is capturing silence
(wrong device). If VAD never fires, the audio may be too quiet or
the threshold too high.
"""

from __future__ import annotations

import math
import struct
import sys
import time

def main() -> None:
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        print("ERROR: PyAudioWPatch not installed. Run: pip install PyAudioWPatch")
        sys.exit(1)

    pa = pyaudio.PyAudio()

    # --- Step 1: Find the loopback device ---
    print("=" * 60)
    print("STEP 1: Finding WASAPI loopback device...")
    print("=" * 60)

    try:
        wasapi_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        print("ERROR: WASAPI not available on this system.")
        pa.terminate()
        sys.exit(1)

    # Find default output device and its loopback
    default_out_idx = wasapi_info.get("defaultOutputDevice", -1)
    if default_out_idx is None or default_out_idx < 0:
        print("ERROR: No default output device found.")
        pa.terminate()
        sys.exit(1)

    out_dev = pa.get_device_info_by_index(default_out_idx)
    target_name = out_dev.get("name", "")
    print(f"  Default output device: {target_name} (index {default_out_idx})")

    # Search for matching loopback
    loopback_dev = None
    print("\n  Available loopback devices:")
    for i in range(pa.get_device_count()):
        dev = pa.get_device_info_by_index(i)
        if dev.get("maxInputChannels", 0) > 0 and "[Loopback]" in dev.get("name", ""):
            marker = ""
            if target_name.lower() in dev["name"].lower():
                loopback_dev = dev
                marker = " <-- MATCH"
            print(f"    [{i}] {dev['name']}{marker}")

    if loopback_dev is None:
        print("\nERROR: Could not find a loopback device matching the default output.")
        print("Falling back to the first available loopback device...")
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if dev.get("maxInputChannels", 0) > 0 and "[Loopback]" in dev.get("name", ""):
                loopback_dev = dev
                break

    if loopback_dev is None:
        print("FATAL: No loopback device found at all.")
        pa.terminate()
        sys.exit(1)

    device_rate = int(loopback_dev["defaultSampleRate"])
    device_channels = int(loopback_dev["maxInputChannels"])
    print(f"\n  Selected loopback: {loopback_dev['name']}")
    print(f"  Sample rate: {device_rate} Hz, Channels: {device_channels}")

    # --- Step 2: Capture audio ---
    print("\n" + "=" * 60)
    print("STEP 2: Capturing 5 seconds of audio...")
    print("  (Play some audio/video in English now!)")
    print("=" * 60)

    frames_per_buffer = int(device_rate * 0.032)  # 32ms chunks
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=device_channels,
        rate=device_rate,
        input=True,
        input_device_index=int(loopback_dev["index"]),
        frames_per_buffer=frames_per_buffer,
    )
    stream.start_stream()

    capture_duration = 5.0
    start = time.monotonic()
    all_rms = []
    total_chunks = 0

    while time.monotonic() - start < capture_duration:
        raw = stream.read(frames_per_buffer, exception_on_overflow=False)

        # Convert to mono
        if device_channels > 1:
            n_frames = len(raw) // (device_channels * 2)
            mono_samples = []
            for i in range(n_frames):
                total = 0
                for ch in range(device_channels):
                    offset = (i * device_channels + ch) * 2
                    (s,) = struct.unpack_from("<h", raw, offset)
                    total += s
                mono_samples.append(total // device_channels)
            mono_data = struct.pack(f"<{len(mono_samples)}h", *mono_samples)
        else:
            mono_data = raw

        # RMS
        n_samples = len(mono_data) // 2
        samples = struct.unpack(f"<{n_samples}h", mono_data)
        sum_sq = sum(s * s for s in samples)
        rms = math.sqrt(sum_sq / max(n_samples, 1)) / 32768.0
        all_rms.append(rms)
        total_chunks += 1

    stream.stop_stream()
    stream.close()

    avg_rms = sum(all_rms) / len(all_rms) if all_rms else 0
    max_rms = max(all_rms) if all_rms else 0
    avg_db = 20 * math.log10(max(avg_rms, 1e-10))
    max_db = 20 * math.log10(max(max_rms, 1e-10))

    print(f"\n  Captured {total_chunks} chunks in {capture_duration}s")
    print(f"  Average RMS: {avg_rms:.6f} ({avg_db:.1f} dB)")
    print(f"  Peak RMS:    {max_rms:.6f} ({max_db:.1f} dB)")

    if avg_rms < 0.001:
        print("\n  ⚠️  WARNING: Audio level is extremely low (near silence).")
        print("  This likely means you're capturing from the WRONG device,")
        print("  or no audio is playing on your system.")
    elif avg_rms < 0.01:
        print("\n  ⚠️  Audio level is low but present. VAD may struggle.")
    else:
        print("\n  ✅ Audio level looks healthy!")

    # --- Step 3: Test VAD ---
    print("\n" + "=" * 60)
    print("STEP 3: Testing Silero VAD on captured audio...")
    print("=" * 60)

    try:
        import torch
        from silero_vad import load_silero_vad
    except ImportError:
        print("  ⚠️  silero-vad or torch not installed. Skipping VAD test.")
        pa.terminate()
        return

    model = load_silero_vad()

    # Re-capture with VAD
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=device_channels,
        rate=device_rate,
        input=True,
        input_device_index=int(loopback_dev["index"]),
        frames_per_buffer=int(device_rate * 0.032),
    )
    stream.start_stream()

    print("  Listening for 5 more seconds (play audio now!)...\n")
    start = time.monotonic()
    speech_count = 0
    total_vad_chunks = 0
    confidences = []

    while time.monotonic() - start < 5.0:
        raw = stream.read(int(device_rate * 0.032), exception_on_overflow=False)

        # To mono
        if device_channels > 1:
            n_frames = len(raw) // (device_channels * 2)
            mono_samples = []
            for i in range(n_frames):
                total = 0
                for ch in range(device_channels):
                    offset = (i * device_channels + ch) * 2
                    (s,) = struct.unpack_from("<h", raw, offset)
                    total += s
                mono_samples.append(total // device_channels)
        else:
            n_samples_raw = len(raw) // 2
            mono_samples = list(struct.unpack(f"<{n_samples_raw}h", raw))

        # Resample to 16kHz if needed
        if device_rate != 16000:
            ratio = device_rate / 16000
            new_len = int(len(mono_samples) / ratio)
            resampled = []
            for i in range(new_len):
                src = i * ratio
                idx = int(src)
                if idx < len(mono_samples):
                    resampled.append(mono_samples[idx])
            mono_samples = resampled

        # To float
        float_samples = [s / 32768.0 for s in mono_samples]
        tensor = torch.FloatTensor(float_samples)

        confidence = model(tensor, 16000).item()
        confidences.append(confidence)
        total_vad_chunks += 1

        if confidence >= 0.5:
            speech_count += 1

    stream.stop_stream()
    stream.close()
    pa.terminate()

    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    max_conf = max(confidences) if confidences else 0

    print(f"  Total chunks: {total_vad_chunks}")
    print(f"  Speech detections (conf >= 0.5): {speech_count}/{total_vad_chunks}")
    print(f"  Average confidence: {avg_conf:.4f}")
    print(f"  Peak confidence:    {max_conf:.4f}")

    if speech_count == 0:
        print("\n  ❌ VAD detected NO speech at all.")
        print("  Possible causes:")
        print("    1. No audio is playing on the system")
        print("    2. Wrong loopback device (see Step 1)")
        print("    3. VAD threshold too high (currently 0.5)")
    elif speech_count < total_vad_chunks * 0.1:
        print(f"\n  ⚠️  VAD detected speech in only {speech_count} chunks.")
        print("  The audio may be too quiet or intermittent.")
    else:
        print(f"\n  ✅ VAD is detecting speech! ({speech_count}/{total_vad_chunks} chunks)")
        print("  The pipeline should produce subtitles when running.")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
