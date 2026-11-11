from unittest.mock import patch

from ..constants.dmd import FileTypes
from ..constants.lambdas import (
    L2_SCIENCE_EXPOSURE_CONFIG,
    MessageKeys,
)
from ..lambdas.ingest import create_table_class_from_message


def test_create_table_class_from_message_uses_exposure_status_config():
    message = {
        MessageKeys.FILE_TYPE: FileTypes.L2_SCIENCE,
        MessageKeys.EXPOSURE_TYPE: None,
        MessageKeys.FILENAME: "test.fits",
        MessageKeys.ARCHIVE_BUCKET: "test-bucket",
        MessageKeys.ARCHIVE_OBJECT_KEY: "test.fits",
        MessageKeys.FILE_CREATION_TIMESTAMP: "2026-01-02T03:04:05.000Z",
    }
    observation_info = {
        "program_num": 1,
        "execution_num": 2,
        "pass_num": 3,
        "segment_num": 4,
        "obs_num": 5,
        "visit_num": 6,
        "exposure_num": 7,
        "visit_id": "visit-id",
        "detector": "detector",
        "optical_element": "filter",
    }

    with patch(
        "rdmt_spire.lambdas.ingest.get_info_from_filename",
        return_value=observation_info,
    ):
        for exposure_type, config in L2_SCIENCE_EXPOSURE_CONFIG.items():
            message[MessageKeys.EXPOSURE_TYPE] = exposure_type
            meta_table = create_table_class_from_message(message)

            assert meta_table.exposure_type == config["exposure_type"]
            for status_column, status_value in config["status_values"].items():
                assert getattr(meta_table, status_column) == int(status_value)