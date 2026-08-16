# This directory holds sample audio fixtures for testing.
# Place .wav files here for integration tests.
# A sample_audio.wav can be generated with:
#   python -c "
#   import wave, struct, math
#   with wave.open('tests/fixtures/sample_audio.wav', 'w') as w:
#       w.setnchannels(1)
#       w.setsampwidth(2)
#       w.setframerate(16000)
#       for i in range(16000 * 5):  # 5 seconds
#           sample = int(math.sin(2 * math.pi * 440 * i / 16000) * 16000)
#           w.writeframes(struct.pack('<h', sample))
#   "
