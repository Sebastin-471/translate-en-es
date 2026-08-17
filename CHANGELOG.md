# Changelog

## [1.0.0] - 2026-08-17

¡Bienvenido a la primera versión funcional de **Live Translator**! En esta actualización hemos conectado el motor real de Inteligencia Artificial y hemos construido una interfaz gráfica completa para que puedas usarlo sin tocar código.

### ✨ Nuevas Características
- **Traducción con IA Real:** El sistema ahora transcribe audio (Whisper) y lo traduce al español (MarianMT) en tiempo real usando todo el poder de tu tarjeta gráfica.
- **Bandeja del Sistema (System Tray):** El programa ahora vive discretamente como un icono "EN-ES" junto a la hora y batería de Windows. Puedes hacer clic derecho en él para pausar la traducción, abrir ajustes o cerrar la app.
- **Panel de Configuración Visual:** ¡Adiós a los archivos `.yaml`! Ahora tienes una ventana con estilo moderno (modo oscuro) donde puedes cambiar el tamaño de los subtítulos, elegir el modelo de IA y ajustar tu dispositivo de audio con simples clics.
- **Feedback de Inicio:** Al abrir la aplicación, un mensaje en pantalla te confirmará que el traductor está activo y escuchando, desapareciendo inteligentemente a los 8 segundos.

### 🔧 Mejoras
- **Buffers de Audio:** Ajustamos la captura de audio a ventanas de 32 milisegundos para cumplir con los estrictos requerimientos del detector de voz, optimizando la latencia.
- **Detección de Hardware:** Mejor distribución de memoria (VRAM) para tarjetas NVIDIA, detectando correctamente la capacidad de equipos como la RTX 3060.

### 🐛 Corrección de Errores
- **Audio Fantasma Reparado:** Corregimos un problema donde el traductor se quedaba escuchando canales virtuales en silencio (como *Voicemeeter*). Ahora se ancla automáticamente al audio de tus auriculares reales.

---
## [0.1.0] - 2026-08-16

### ✨ New Features
- **Real-time Pipeline**: Core architecture for capturing system audio, running Voice Activity Detection (VAD), Automatic Speech Recognition (ASR), and Machine Translation (MT) in real-time.
- **Always-on-top UI**: Configurable Tkinter-based translucent overlay for displaying translated subtitles.
- **Platform Support**: Modular audio capture backend supporting Windows (WASAPI Loopback) and Linux (PipeWire).
- **GPU Resource Management**: Centralized VRAM budgeting to support graceful degradation on limited hardware (e.g., RTX 3060 12GB).
- **Mock Engines**: Complete suite of mock engines allowing development and testing without requiring heavy ML models or GPU access.
- **Global Hotkeys**: Cross-platform global hotkey management using `pynput` for toggling UI and pipeline state.

### 🔧 Improvements
- **Typed Asynchronous Architecture**: Fully decoupled pipeline using Python `typing.Protocol`, `dataclasses`, and `asyncio.Queue`.
- **Structured Logging**: Deep observability with ISO 8601 timestamps and latency metrics tracking (p50, p95, p99) per pipeline stage.
- **Config Management**: YAML-based configuration with environment variable override support.

### 🐛 Fixes
- Initial Release.
