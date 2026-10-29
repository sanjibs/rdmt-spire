from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..constants.database import (
    ARCHIVE_LENGTH,
    DETECTOR_LENGTH,
    FILENAME_LENGTH,
    OPTICAL_ELEMENT_LENGTH,
    SOFTWARE_VERSION_LENGTH,
    VISIT_ID_LENGTH,
    MonitorStatus,
)
from ..constants.dmd import FileTypes
from .base import Base, ResultsBase


class L1GuideWindowMetaTable(Base):
    """Class containing schema for the L1 Guide Window data metadata table."""

    __tablename__ = "l1_guide_window_meta"
    # Information from the DMD SNS notification
    filename:               Mapped[str] = mapped_column(String(FILENAME_LENGTH), primary_key=True) # body['filename']
    reprocess_number:       Mapped[int] = mapped_column(Integer(), primary_key=True) # determined from body['reprocessingState]
    # 0 for prompt (reprocessingState = None), 
    # +1 for each reprocessing (reprocessingState = 'reprocessed' or 'data_release')
    program_number:         Mapped[int] = mapped_column(Integer()) # from filename
    visit_id:               Mapped[str] = mapped_column(String(VISIT_ID_LENGTH)) # from filename
    gw_acquisition_number:  Mapped[int] = mapped_column(Integer()) # from filename
    acquisition_id:         Mapped[str] = mapped_column(String(VISIT_ID_LENGTH+2)) # from filename    
    detector:               Mapped[str] = mapped_column(String(DETECTOR_LENGTH)) # from filename
    optical_element:        Mapped[str] = mapped_column(String(OPTICAL_ELEMENT_LENGTH)) # from filename
    archive_bucket:         Mapped[str] = mapped_column(String(ARCHIVE_LENGTH)) # from body['archiveBucket']
    archive_key:            Mapped[str] = mapped_column(String(ARCHIVE_LENGTH)) # from body['archiveObjectKey']
    file_created_datetime:  Mapped[datetime] = mapped_column(DateTime()) # from body['fileCreationTimestamp']
    dmd_notify_datetime:    Mapped[datetime] = mapped_column(DateTime()) # from outer message structure

    # Information that requires opening the file to extract
    acq_start_datetime:     Mapped[Optional[datetime]] = mapped_column(DateTime()) # from meta.start_time
    sdf_version:            Mapped[Optional[str]] = mapped_column(String(SOFTWARE_VERSION_LENGTH)) # from meta.sdf_software_version

    # Information populated by RDMT
    monitor_end_datetime:   Mapped[Optional[datetime]] = mapped_column(DateTime()) # populated by Spire
    # Below are the monitor completion chec()ks:
    # -1 indicates the monitor should not be run
    # 0 indicates the monitor still needs to be run
    # 1 indicates the monitor ran successfully
    guide_window_status:         Mapped[int] = mapped_column(Integer(), default=int(MonitorStatus.NEEDS_TO_RUN)) # populated by Spire

    def _get_verification_columns(self):
        return [
            "filename",
            "reprocess_number",
            "program_number",
            "gw_acquisition_number",
            "acquisition_id",
            "acq_start_datetime",
            "visit_id",
            "detector",
            "optical_element",
            "archive_bucket",
            "archive_key",
            "file_created_datetime",
            "dmd_notify_datetime",
            "sdf_version",
            "monitor_end_datetime",
            "guide_window_status",
        ]
    
    # Used for mapping the table classes to the file types they relate to
    file_type = FileTypes.L1_GUIDE_WINDOW


class L1GuideWindowResultsTable(ResultsBase):
    """Class containing schema for the L1 Guide Window data monitoring results table."""

    __tablename__ = "l1_guide_window_results"
    filename:                   Mapped[str] = mapped_column(String(FILENAME_LENGTH), primary_key=True) # from DMD notification
    reprocess_number:           Mapped[int] = mapped_column(Integer(), primary_key=True) # from DMD notification
    # 0 for prompt (reprocessingState = None), 
    # +1 for each reprocessing (reprocessingState = 'reprocessed' or 'data_release') reprocessing

    acquisition_status:          Mapped[Optional[str]] = mapped_column(String(30))
    acquisition_status_eval:     Mapped[Optional[bool]] = mapped_column(Boolean())
    saturation_status:           Mapped[Optional[str]] = mapped_column(String(20))
    saturation_status_eval:      Mapped[Optional[bool]] = mapped_column(Boolean())

    median_background:           Mapped[Optional[float]] = mapped_column(Float())
    median_background_eval:      Mapped[Optional[bool]] = mapped_column(Boolean())
    mean_noise:                  Mapped[Optional[float]] = mapped_column(Float())
    mean_noise_eval:             Mapped[Optional[bool]] = mapped_column(Boolean())
    median_background_std:       Mapped[Optional[float]] = mapped_column(Float())
    median_background_std_eval:  Mapped[Optional[bool]] = mapped_column(Boolean())
    noise_std:                   Mapped[Optional[float]] = mapped_column(Float())
    noise_std_eval:              Mapped[Optional[bool]] = mapped_column(Boolean())

    num_background_outliers:        Mapped[Optional[int]] = mapped_column(Integer())
    num_background_outliers_eval:   Mapped[Optional[bool]] = mapped_column(Boolean())
    num_noise_outliers:             Mapped[Optional[int]] = mapped_column(Integer())
    num_noise_outliers_eval:        Mapped[Optional[bool]] = mapped_column(Boolean())

    mean_count_rate:                Mapped[Optional[float]] = mapped_column(Float())
    mean_count_rate_eval:           Mapped[Optional[bool]] = mapped_column(Boolean())
    std_count_rate:                 Mapped[Optional[float]] = mapped_column(Float())
    std_count_rate_eval:            Mapped[Optional[bool]] = mapped_column(Boolean())
    num_count_rate_outliers:        Mapped[Optional[int]] = mapped_column(Integer())
    num_count_rate_outliers_eval:   Mapped[Optional[bool]] = mapped_column(Boolean())

    rms_x_centroid_error:        Mapped[Optional[float]] = mapped_column(Float())
    rms_x_centroid_error_eval:   Mapped[Optional[bool]] = mapped_column(Boolean())
    rms_y_centroid_error:        Mapped[Optional[float]] = mapped_column(Float()) 
    rms_y_centroid_error_eval:   Mapped[Optional[bool]] = mapped_column(Boolean())

    rms_centroid_offset:         Mapped[Optional[float]] = mapped_column(Float())
    rms_centroid_offset_eval:    Mapped[Optional[bool]] = mapped_column(Boolean()) 

    def _get_verification_columns(self):
        return [
            "filename",
            "reprocess_number",
            "acquisition_status",
            "acquisition_status_eval",
            "saturation_status",
            "saturation_status_eval",
            "median_background",
            "median_background_eval",
            "mean_noise",
            "mean_noise_eval",
            "median_background_std",
            "median_background_std_eval",
            "noise_std",
            "noise_std_eval",
            "num_background_outliers",
            "num_background_outliers_eval",
            "num_noise_outliers",
            "num_noise_outliers_eval",
            "mean_count_rate",
            "mean_count_rate_eval",
            "std_count_rate",
            "std_count_rate_eval",
            "num_count_rate_outliers",
            "num_count_rate_outliers_eval",
            "rms_x_centroid_error",
            "rms_x_centroid_error_eval",
            "rms_y_centroid_error",
            "rms_y_centroid_error_eval",
            "rms_centroid_offset",
            "rms_centroid_offset_eval",
        ]

    # Used for mapping the table classes to the file types they relate to
    file_type = FileTypes.L1_GUIDE_WINDOW