"""Configuration constants for satellite image processing."""
from enum import Enum

# Memory management
MAX_WINDOW_SIZE_GB = 1.0  # Maximum window size in GB before chunking
CHUNK_HEIGHT = 10000  # Number of rows to read per chunk

# CRS
class CRS(Enum):
    WGS84 = "EPSG:4326"
    WEB_MERCATOR = "EPSG:3857"
    UTM_32N = "EPSG:32632"
    UTM_33N = "EPSG:32633"
    UTM_34N = "EPSG:32634"
    UTM_35N = "EPSG:32635"
    UTM_36N = "EPSG:32636"
    UTM_37N = "EPSG:32637"
    UTM_38N = "EPSG:32638"
    UTM_39N = "EPSG:32639"

    def __int__(self):
        return int(self.value[self.value.find(":") + 1:])