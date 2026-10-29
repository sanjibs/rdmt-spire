import os
from enum import StrEnum

from .database import MonitorStatus
from .dmd import ExposureTypes

# AWS Parameter store path and parameter names
prefix = os.getenv("PROFILE_NAME_PREFIX", "rdmt")
AWS_PARAMETER_PATH = f"/{prefix}/config/"

# SQS queue parameter names
ASTROMETRY_MONITOR_QUEUE = "astrometry_monitor_queue"
ESSENTIAL_L2_MONITOR_QUEUE = "essential_l2_monitor_queue"
GUIDE_WINDOW_MONITOR_QUEUE = "guide_window_monitor_queue"
SOURCE_CATALOG_MONITOR_QUEUE = "source_catalog_monitor_queue"

AWS_MONITOR_QUEUES = [
    ASTROMETRY_MONITOR_QUEUE,
    ESSENTIAL_L2_MONITOR_QUEUE,
    GUIDE_WINDOW_MONITOR_QUEUE,
    SOURCE_CATALOG_MONITOR_QUEUE,
]

# SNS topic parameter names
SCIENCE_REPORTING_TOPIC = "science_reporting_topic"
GUIDE_WINDOW_REPORTING_TOPIC = "guide_window_reporting_topic"

AWS_SNS_TOPICS = [
    SCIENCE_REPORTING_TOPIC,
    GUIDE_WINDOW_REPORTING_TOPIC,
]

# S3 bucket parameter names
PARQUET_FILE_BUCKET = "parquet_file_bucket"
ALEMBIC_VERSIONS_BUCKET = "alembic_versions_bucket"
ASTROMETRY_MONITOR_DATA_BUCKET = "astrometry_monitor_data_bucket"
SOURCE_CATALOG_MONITOR_DATA_BUCKET = "source_catalog_monitor_data_bucket"

AWS_S3_BUCKETS = [
    PARQUET_FILE_BUCKET,
    ALEMBIC_VERSIONS_BUCKET,
    ASTROMETRY_MONITOR_DATA_BUCKET,
    SOURCE_CATALOG_MONITOR_DATA_BUCKET,
]

# Database parameter names
DB_NAME = "db_name"
DB_SECRET_NAME = "db_secret_name"

AWS_DBS = [
    DB_NAME,
    DB_SECRET_NAME,
]

# Define valid exposure type strings for within Spire itself
WIM_EXPOSURE = "WIM"
WSM_EXPOSURE = "WSM"
FLAT_EXPOSURE = "FLAT"

# Initial monitor status policy for each DMD L2 science exposure type.
# Add future monitor ``*_status`` columns to each exposure's ``status_values``.
L2_SCIENCE_EXPOSURE_CONFIG = {
    ExposureTypes.WFI_IMAGE: {
        "exposure_type": WIM_EXPOSURE,
        "status_values": {
            "astrometry_status": MonitorStatus.PENDING_CHECK,
            "source_catalog_status": MonitorStatus.PENDING_CHECK,
        },
    },
    ExposureTypes.WFI_SPECTRAL: {
        "exposure_type": WSM_EXPOSURE,
        "status_values": {
            "astrometry_status": MonitorStatus.NOT_APPLICABLE,
            "source_catalog_status": MonitorStatus.NOT_APPLICABLE,
        },
    },
    ExposureTypes.WFI_FLAT: {
        "exposure_type": FLAT_EXPOSURE,
        "status_values": {
            "astrometry_status": MonitorStatus.NOT_APPLICABLE,
            "source_catalog_status": MonitorStatus.NOT_APPLICABLE,
        },
    },
}

#Currently we do not need to distinguish WFI_GRISM and WFI_PRISM from WFI_SPECTRAL in the exposure config.
L2_SCIENCE_EXPOSURE_CONFIG[ExposureTypes.WFI_GRISM] = L2_SCIENCE_EXPOSURE_CONFIG[ExposureTypes.WFI_SPECTRAL].copy()
L2_SCIENCE_EXPOSURE_CONFIG[ExposureTypes.WFI_PRISM] = L2_SCIENCE_EXPOSURE_CONFIG[ExposureTypes.WFI_SPECTRAL].copy()


VALID_L2_SCIENCE_EXPOSURE_TYPES = tuple(L2_SCIENCE_EXPOSURE_CONFIG)

# Retain the exposure group constants as derived views of the configuration.
L2_SCIENCE_WIM_EXPOSURE_TYPES = [
    exposure_type for exposure_type, config in L2_SCIENCE_EXPOSURE_CONFIG.items()
    if config["exposure_type"] == WIM_EXPOSURE
]
L2_SCIENCE_WSM_EXPOSURE_TYPES = [
    exposure_type for exposure_type, config in L2_SCIENCE_EXPOSURE_CONFIG.items()
    if config["exposure_type"] == WSM_EXPOSURE
]
L2_SCIENCE_FLAT_EXPOSURE_TYPES = [
    exposure_type for exposure_type, config in L2_SCIENCE_EXPOSURE_CONFIG.items()
    if config["exposure_type"] == FLAT_EXPOSURE
]


class FunctionTypes(StrEnum):
    MONITOR = "monitor"
    METADATA_CHECK = "meta_check"
    REPORT_GEN = "report"

class MessageKeys(StrEnum):
    # The following keys are defined by DMD in their Archiving SNS Topic notifications
    ARCHIVE_BUCKET = "archiveBucket"
    ARCHIVE_OBJECT_KEY = "archiveObjectKey"
    FILE_TYPE = "fileType"
    FILE_CREATION_TIMESTAMP = "fileCreationTimestamp"
    FILENAME = "filename"
    REPROCESSING_STATE = "reprocessingState"
    EXPOSURE_TYPE = "exposureType"

    # The following keys are only used within Spire itself
    FUNCTION_TYPE = "function_type"
    MONITOR_NAME = "monitor_name"
    REPROCESS_NUMBER = "reprocess_number"

    METADATA_CHECK_TYPE = 'metadata_check_type'
    REPORT_TYPE = 'report_type'

# Monitor names for use within Spire itself. These are only then names of monitors triggered by lambda functions
class MonitorGroupName(StrEnum):
    ESSENTIAL_WIM = "essential_wim"
    ESSENTIAL_WSM = "essential_wsm"
    ESSENTIAL_FLAT = "essential_flat"
    ESSENTIAL_GW = "essential_guide_window"
    ASTROMETRY = "astrometry"
    SOURCE_CATALOG = "source_catalog"

ESSENTIAL_MONITOR_FROM_EXPOSURE_TYPE = {
    WIM_EXPOSURE: MonitorGroupName.ESSENTIAL_WIM,
    WSM_EXPOSURE: MonitorGroupName.ESSENTIAL_WSM,
    FLAT_EXPOSURE: MonitorGroupName.ESSENTIAL_FLAT,
}