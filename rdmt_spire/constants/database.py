# Define all the fixed string lengths for the database column
# These numbers were taken from the rad repo on 12/18/25 (before v0.29.0 release)
from enum import IntEnum

FILENAME_LENGTH = 120
VISIT_ID_LENGTH = 19
OBSERVATION_ID_LENGTH = 28
DETECTOR_LENGTH = 5
OPTICAL_ELEMENT_LENGTH = 4
SOFTWARE_VERSION_LENGTH = 120
ARCHIVE_LENGTH = 120
DATA_RELEASE_ID_LENGTH = 16
EXPOSURE_TYPE_LENGTH = 16 # This has plenty of buffer for future exposure types

class MonitorStatus(IntEnum):
    """Integer values stored in monitor ``*_status`` columns."""
    PENDING_CHECK = -2
    NOT_APPLICABLE = -1
    NEEDS_TO_RUN = 0
    COMPLETE = 1