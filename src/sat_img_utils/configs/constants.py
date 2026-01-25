"""Configuration constants for satellite image processing."""

# Building detection thresholds
BUILDINGS_THRESHOLD = 15  # Threshold for GHSL building detection

# Memory management
MAX_WINDOW_SIZE_GB = 1.0  # Maximum window size in GB before chunking
CHUNK_HEIGHT = 10000  # Number of rows to read per chunk
