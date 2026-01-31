# CAPELLA CONSTANTS

# Polarizations
# define an enumeration for polarizations
from enum import Enum

class CapellaPolarization(Enum):
    HH = "HH"
    HV = "HV"
    VH = "VH"
    VV = "VV"

# these constants were set in an extremely hacky manner based on visual inspection
# in the future, we may want to derive these more systematically
class CapellaPercentValue(Enum):
    LO_HH_VV = 0.0009
    LO_HV_VH = 32.5
    HI_HH_VV = 99.99999
    HI_HV_VH = 75.0

CAPELLA_BANDS = 1
CAPELLA_EXTRA_CTX = {
    "threshold_fraction_filter_eq": { 
        "filter_value": 0.0,
        "nodata": None, 
    "upper": True,
        "fraction_value": 0.75
    }, 
    "min_land_fraction_filter_random": {
        "land_mask": None,  # to be set per-tile
        "min_land_threshold": 0.1,
        "discard_prob": 0.9,
    } 
}