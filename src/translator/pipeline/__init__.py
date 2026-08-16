"""Pipeline orchestration layer.

Imports from `translator.core` ONLY. Connects pipeline stages with typed
async queues and manages the full lifecycle (start → run → graceful shutdown).
"""
