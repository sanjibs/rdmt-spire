from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..constants import photoastro_constants, source_catalog_constants
from ..constants.database import (
    ARCHIVE_LENGTH,
    DATA_RELEASE_ID_LENGTH,
    DETECTOR_LENGTH,
    FILENAME_LENGTH,
    OBSERVATION_ID_LENGTH,
    OPTICAL_ELEMENT_LENGTH,
    SOFTWARE_VERSION_LENGTH,
    VISIT_ID_LENGTH,
)
from ..constants.dmd import FileTypes
from ..utilities.property import Property
from .base import Base, ResultsBase


class L2ScienceMetaTable(Base):
    """Class containing schema for the L2 Science data metadata table."""

    __tablename__ = "l2_science_meta"
    # Information from the DMD SNS notification
    filename:               Mapped[str] = mapped_column(String(FILENAME_LENGTH), primary_key=True) # body['filename']
    reprocess_number:       Mapped[int] = mapped_column(Integer(), primary_key=True) # determined from body['reprocessingState]
    # 0 for prompt (reprocessingState = None), 
    # +1 for each reprocessing (reprocessingState = 'reprocessed' or 'data_release')
    program_number:         Mapped[int] = mapped_column(Integer()) # from filename
    execution_number:       Mapped[int] = mapped_column(Integer()) # from filename
    pass_number:            Mapped[int] = mapped_column(Integer()) # from filename
    segment_number:         Mapped[int] = mapped_column(Integer()) # from filename
    observation_number:     Mapped[int] = mapped_column(Integer()) # from filename
    visit_number:           Mapped[int] = mapped_column(Integer()) # from filename
    exposure_number:        Mapped[int] = mapped_column(Integer()) # from filename
    visit_id:               Mapped[str] = mapped_column(String(VISIT_ID_LENGTH)) # from filename
    detector:               Mapped[str] = mapped_column(String(DETECTOR_LENGTH)) # from filename
    optical_element:        Mapped[str] = mapped_column(String(OPTICAL_ELEMENT_LENGTH)) # from filename
    archive_bucket:         Mapped[str] = mapped_column(String(ARCHIVE_LENGTH)) # from body['archiveBucket']
    archive_key:            Mapped[str] = mapped_column(String(ARCHIVE_LENGTH)) # from body['archiveObjectKey']
    file_created_datetime:  Mapped[datetime] = mapped_column(DateTime()) # from body['fileCreationTimestamp']
    dmd_notify_datetime:    Mapped[datetime] = mapped_column(DateTime()) # from outer message structure
    data_release_id:        Mapped[Optional[str]] = mapped_column(String(DATA_RELEASE_ID_LENGTH))

    # Information that requires opening the file to extract
    observation_id:         Mapped[Optional[str]] = mapped_column(String(OBSERVATION_ID_LENGTH)) # from meta.observation.observation_id
    exp_start_datetime:     Mapped[Optional[datetime]] = mapped_column(DateTime(), index=True) # from meta.exposure.start_time
    romancal_version:       Mapped[Optional[str]] = mapped_column(String(SOFTWARE_VERSION_LENGTH)) # from meta.calibration_software_version
    crds_context:           Mapped[Optional[str]] = mapped_column(String(SOFTWARE_VERSION_LENGTH)) # from meta.ref_file.crds.context
    sdf_version:            Mapped[Optional[str]] = mapped_column(String(SOFTWARE_VERSION_LENGTH)) # from meta.sdf_software_version

    # Information populated by RDMT
    monitor_end_datetime:   Mapped[Optional[datetime]] = mapped_column(DateTime()) # populated by Spire
    # Below are the monitor completion checks:
    # 1 indicates the monitor ran successfully
    # 0 indicates the monitor still needs to be run
    # -1 indicates the monitor should not be run
    # -2 indicates the monitor may need to be run depending on the metadata checks
    essential_status:       Mapped[int] = mapped_column(Integer(), default=0) # populated by Spire
    
    # periodic monitors are run on a schedule and may not be run for every file. The status is set to -1 if the monitor should not be run for this file.   
    astrometry_status:      Mapped[int] = mapped_column(Integer(), default=-2) # populated by Spire
    source_catalog_status:  Mapped[int] = mapped_column(Integer(), default=-2) # populated by Spire

    def _get_verification_columns(self):
        return [
            "filename",
            "reprocess_number",
            "program_number",
            "exposure_number",
            "visit_id",
            "detector",
            "optical_element",
            "archive_bucket",
            "archive_key",
            "file_created_datetime",
            "dmd_notify_datetime",
            "data_release_id",
            "observation_id",
            "exp_start_datetime",
            "romancal_version",
            "crds_context",
            "sdf_version",
            "monitor_end_datetime",
            "essential_status",
            "astrometry_status",
            "source_catalog_status",
        ]
    
    # Used for mapping the table classes to the file types they relate to
    file_type = FileTypes.L2_SCIENCE


def _add_science_results_columns(cls):
    # This function dynamically adds columns to the class 
    # for science results based on the properties defined in monitors 

    # We start with source_catalog monitor properties 
    # One can add properties from other monitors if need be 
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    for monitor_constants in [source_catalog_constants, photoastro_constants]:

        for prop_name, prop_unit, stat_style, outlier_thresholds in monitor_constants.PROPERTIES:
            for suffix1 in monitor_constants.SUFFIX1:
                property = Property(prop_name, stat_style, suffix1=suffix1, outlier_thresholds=outlier_thresholds)
                property.compute(x, prop_unit)
                for card in property.cards.values():
                    if card.data_name not in cls.__table__.c:
                        if type(card.data_value) is int:
                            col_type = Integer()
                            py_type = Optional[int]
                        else:
                            col_type = Float()
                            py_type = Optional[float]

                        # Construct the column name and add it to the class annotations and mapped columns
                        cls.__annotations__[card.data_name] = Mapped[py_type]
                        setattr(cls, card.data_name, mapped_column(col_type))

                        # Construct the evaluation column name and add it to the class annotations and mapped columns
                        cls.__annotations__[card.data_name+'_eval'] = Mapped[Optional[bool]]
                        setattr(cls, card.data_name+'_eval', mapped_column(Boolean()))
                    else:
                        raise ValueError(f"Column {card.data_name} already exists in the table.")

    return cls

@_add_science_results_columns
class L2ScienceResultsTable(ResultsBase):
    """Class containing schema for the L2 Science data monitoring results table."""

    __tablename__ = "l2_science_results"
    filename:                   Mapped[str] = mapped_column(String(FILENAME_LENGTH), primary_key=True) # from DMD notification
    reprocess_number:           Mapped[int] = mapped_column(Integer(), primary_key=True) # from DMD notification
    # 0 for prompt (reprocessingState = None), 
    # +1 for each reprocessing (reprocessingState = 'reprocessed' or 'data_release') reprocessing

    # Astrometry monitor results
    astrometric_offset:         Mapped[Optional[float]] = mapped_column(Float())
    astrometric_offset_eval:    Mapped[Optional[bool]] = mapped_column(Boolean())
    num_astrometric_sources:        Mapped[Optional[int]] = mapped_column(Integer())
    num_astrometric_sources_eval:   Mapped[Optional[bool]] = mapped_column(Boolean())

    # 1/f Noise monitor results
    noise_1f_power_ratio:       Mapped[Optional[float]] = mapped_column(Float())
    noise_1f_power_ratio_eval:  Mapped[Optional[bool]] = mapped_column(Boolean())

    # Pixel monitor results
    plate_scale_center_x:        Mapped[Optional[float]] = mapped_column(Float())
    plate_scale_center_x_eval:   Mapped[Optional[bool]]  = mapped_column(Boolean())
    plate_scale_center_y:        Mapped[Optional[float]] = mapped_column(Float())
    plate_scale_center_y_eval:   Mapped[Optional[bool]]  = mapped_column(Boolean())

    delta_plate_scale_center:      Mapped[Optional[float]] = mapped_column(Float())
    delta_plate_scale_center_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())
    delta_plate_scale_corner_bl:      Mapped[Optional[float]] = mapped_column(Float())
    delta_plate_scale_corner_bl_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())
    delta_plate_scale_corner_br:      Mapped[Optional[float]] = mapped_column(Float())
    delta_plate_scale_corner_br_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())
    delta_plate_scale_corner_tl:      Mapped[Optional[float]] = mapped_column(Float())
    delta_plate_scale_corner_tl_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())
    delta_plate_scale_corner_tr:      Mapped[Optional[float]] = mapped_column(Float())
    delta_plate_scale_corner_tr_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())

    plate_scale_corner_bl_x:      Mapped[Optional[float]] = mapped_column(Float())
    plate_scale_corner_bl_x_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())
    plate_scale_corner_bl_y:      Mapped[Optional[float]] = mapped_column(Float())
    plate_scale_corner_bl_y_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())
    plate_scale_corner_br_x:      Mapped[Optional[float]] = mapped_column(Float())
    plate_scale_corner_br_x_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())
    plate_scale_corner_br_y:      Mapped[Optional[float]] = mapped_column(Float())
    plate_scale_corner_br_y_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())
    plate_scale_corner_tl_x:      Mapped[Optional[float]] = mapped_column(Float())
    plate_scale_corner_tl_x_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())
    plate_scale_corner_tl_y:      Mapped[Optional[float]] = mapped_column(Float())
    plate_scale_corner_tl_y_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())
    plate_scale_corner_tr_x:      Mapped[Optional[float]] = mapped_column(Float())
    plate_scale_corner_tr_x_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())
    plate_scale_corner_tr_y:      Mapped[Optional[float]] = mapped_column(Float())
    plate_scale_corner_tr_y_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())

    n_saturated_pix:      Mapped[Optional[int]]   = mapped_column(Integer())
    n_saturated_pix_eval:  Mapped[Optional[bool]]  = mapped_column(Boolean())
    min_ramp_value:        Mapped[Optional[float]] = mapped_column(Float())
    min_ramp_value_eval:   Mapped[Optional[bool]]  = mapped_column(Boolean())
    max_ramp_value:        Mapped[Optional[float]] = mapped_column(Float())
    max_ramp_value_eval:   Mapped[Optional[bool]]  = mapped_column(Boolean())
    mean_ramp_value:       Mapped[Optional[float]] = mapped_column(Float())
    mean_ramp_value_eval:  Mapped[Optional[bool]]  = mapped_column(Boolean())
    median_ramp_value:      Mapped[Optional[float]] = mapped_column(Float())
    median_ramp_value_eval: Mapped[Optional[bool]]  = mapped_column(Boolean())
    std_ramp_value:        Mapped[Optional[float]] = mapped_column(Float())
    std_ramp_value_eval:   Mapped[Optional[bool]]  = mapped_column(Boolean())
    p95_ramp_value:        Mapped[Optional[float]] = mapped_column(Float())
    p95_ramp_value_eval:   Mapped[Optional[bool]]  = mapped_column(Boolean())
    p05_ramp_value:        Mapped[Optional[float]] = mapped_column(Float())
    p05_ramp_value_eval:   Mapped[Optional[bool]]  = mapped_column(Boolean())
    

    def _get_verification_columns(self):
        cols = [
            "filename",
            "reprocess_number",
            "astrometric_offset",
            "astrometric_offset_eval",
            "num_astrometric_sources",
            "num_astrometric_sources_eval",
            "noise_1f_power_ratio",
            "noise_1f_power_ratio_eval",
            "plate_scale_center_x",
            "plate_scale_center_x_eval",
            "plate_scale_center_y",
            "plate_scale_center_y_eval",
            "delta_plate_scale_center",
            "delta_plate_scale_center_eval",
            "delta_plate_scale_corner_bl",
            "delta_plate_scale_corner_bl_eval",
            "delta_plate_scale_corner_br",
            "delta_plate_scale_corner_br_eval",
            "delta_plate_scale_corner_tl",
            "delta_plate_scale_corner_tl_eval",
            "delta_plate_scale_corner_tr",
            "delta_plate_scale_corner_tr_eval",
            "plate_scale_corner_bl_x",
            "plate_scale_corner_bl_x_eval",
            "plate_scale_corner_bl_y",
            "plate_scale_corner_bl_y_eval",
            "plate_scale_corner_br_x",
            "plate_scale_corner_br_x_eval",
            "plate_scale_corner_br_y",
            "plate_scale_corner_br_y_eval",
            "plate_scale_corner_tl_x",
            "plate_scale_corner_tl_x_eval",
            "plate_scale_corner_tl_y",
            "plate_scale_corner_tl_y_eval",
            "plate_scale_corner_tr_x",
            "plate_scale_corner_tr_x_eval",
            "plate_scale_corner_tr_y",
            "plate_scale_corner_tr_y_eval",
            "n_saturated_pix",
            "n_saturated_pix_eval",
            "min_ramp_value",
            "min_ramp_value_eval",
            "max_ramp_value",
            "max_ramp_value_eval",
            "mean_ramp_value",
            "mean_ramp_value_eval",
            "median_ramp_value",
            "median_ramp_value_eval",
            "std_ramp_value",
            "std_ramp_value_eval",
            "p95_ramp_value",
            "p95_ramp_value_eval",
            "p05_ramp_value",
            "p05_ramp_value_eval",
        ]

        cols.extend(
            column.name
            for column in self.__table__.columns
            if column.name not in cols
        )

        return cols
    

    # Used for mapping the table classes to the file types they relate to
    file_type = FileTypes.L2_SCIENCE