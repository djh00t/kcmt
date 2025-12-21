from enum import Enum

class AssimilationState(str, Enum):
    OBSERVED = "observed"
    CATALOGUED = "catalogued"
    GOVERNED = "governed"
    CANONICAL = "canonical"
    IPBANKED = "ipbanked"