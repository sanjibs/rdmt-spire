"""Verify parameter selection without accessing AWS or the database."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ..constants.dmd import FileTypes
from ..constants.lambdas import (
    ASTROMETRY_MONITOR_QUEUE,
    AWS_DBS,
    AWS_PARAMETER_PATH,
    ESSENTIAL_L2_MONITOR_QUEUE,
    GUIDE_WINDOW_MONITOR_QUEUE,
    GUIDE_WINDOW_REPORTING_TOPIC,
    PARQUET_FILE_BUCKET,
    REFERENCE_DATA_BUCKET,
    SCIENCE_REPORTING_TOPIC,
    SOURCE_CATALOG_MONITOR_QUEUE,
    MessageKeys,
    MonitorGroupName,
)
from ..lambdas import ingest, metadata_check, monitor, report


@pytest.mark.parametrize(
    "file_type,queue",
    [
        (FileTypes.L2_SCIENCE, ESSENTIAL_L2_MONITOR_QUEUE),
        (FileTypes.L1_GUIDE_WINDOW, GUIDE_WINDOW_MONITOR_QUEUE),
        (FileTypes.L4_SOURCE_CATALOG_L2, None),
    ],
)
def test_ingest_selects_only_relevant_parameters(monkeypatch, file_type, queue):
    fetch = Mock(side_effect=RuntimeError("stop after parameter selection"))
    monkeypatch.setattr(ingest, "fetch_parameters_from_path", fetch)
    monkeypatch.setattr(ingest, "create_table_class_from_message", Mock(return_value=SimpleNamespace()))
    message = {MessageKeys.FILE_TYPE: file_type, MessageKeys.FILENAME: "test_cal.asdf"}

    with pytest.raises(RuntimeError, match="stop after parameter selection"):
        ingest.ingest_single(message, None, "account")

    expected = AWS_DBS + ([queue] if queue is not None else [])
    fetch.assert_called_once_with(AWS_PARAMETER_PATH, expected_parameters=expected)


def test_ingest_batch_only_requests_database_parameters(monkeypatch):
    fetch = Mock(side_effect=RuntimeError("stop after parameter selection"))
    monkeypatch.setattr(ingest, "fetch_parameters_from_path", fetch)

    with pytest.raises(RuntimeError, match="stop after parameter selection"):
        ingest.ingest_batch([], [])

    fetch.assert_called_once_with(AWS_PARAMETER_PATH, expected_parameters=AWS_DBS)


def test_metadata_check_only_requests_supported_queues(monkeypatch):
    fetch = Mock(side_effect=RuntimeError("stop after parameter selection"))
    monkeypatch.setattr(metadata_check, "fetch_parameters_from_path", fetch)

    with pytest.raises(RuntimeError, match="stop after parameter selection"):
        metadata_check.metadata_check_function({}, "account")

    fetch.assert_called_once_with(
        AWS_PARAMETER_PATH,
        expected_parameters=AWS_DBS + [ASTROMETRY_MONITOR_QUEUE, SOURCE_CATALOG_MONITOR_QUEUE],
    )


@pytest.mark.parametrize(
    "monitor_name,extra_parameters",
    [
        (MonitorGroupName.ASTROMETRY, [REFERENCE_DATA_BUCKET]),
        (MonitorGroupName.SOURCE_CATALOG, [REFERENCE_DATA_BUCKET]),
        (MonitorGroupName.ESSENTIAL_WIM, []),
        (MonitorGroupName.ESSENTIAL_GW, []),
    ],
)
def test_monitor_selects_reference_bucket_only_when_needed(monkeypatch, monitor_name, extra_parameters):
    fetch = Mock(side_effect=RuntimeError("stop after parameter selection"))
    monkeypatch.setattr(monitor, "fetch_parameters_from_path", fetch)
    monkeypatch.setattr(monitor, "load_s3_object", Mock())
    monkeypatch.setattr(monitor.asdf, "open", Mock())
    message = {
        MessageKeys.ARCHIVE_BUCKET: "bucket",
        MessageKeys.ARCHIVE_OBJECT_KEY: "key",
        MessageKeys.FILENAME: "test_cal.asdf",
        MessageKeys.MONITOR_NAME: monitor_name,
    }

    with pytest.raises(RuntimeError, match="stop after parameter selection"):
        monitor.monitor_function(message)

    fetch.assert_called_once_with(AWS_PARAMETER_PATH, expected_parameters=AWS_DBS + extra_parameters)


@pytest.mark.parametrize(
    "monitor_name",
    [MonitorGroupName.SOURCE_CATALOG, MonitorGroupName.ASTROMETRY],
)
def test_monitor_config_includes_bundled_calibration_data(monitor_name):
    message = {
        MessageKeys.MONITOR_NAME: monitor_name,
        MessageKeys.L4_ARCHIVE_BUCKET: "l4-bucket",
        MessageKeys.L4_ARCHIVE_OBJECT_KEY: "catalog.asdf",
    }

    config = monitor.generate_monitor_config(message, {REFERENCE_DATA_BUCKET: "reference-bucket"})
    local_data_dir = Path(__file__).resolve().parent.parent / "rdmt_data"

    assert config == {
        monitor_name: {
            "BUNDLED_DATA_DIR": str(local_data_dir),
            "REFERENCE_DATA_DIR": "s3://reference-bucket",
            "L4_ARCHIVE_DIR": "s3://l4-bucket",
            "L4_ARCHIVE_OBJECT_KEY": "catalog.asdf",
        }
    }
    for filename in (
        "zero_points_20260401.ecsv",
        "filter_parameters.ecsv",
        "internal_thermal_backgrounds.ecsv",
        "zodiacal_light.ecsv",
    ):
        assert (Path(config[monitor_name]["BUNDLED_DATA_DIR"]) / "roman_technical_information" / filename).is_file()


@pytest.mark.parametrize(
    "report_type,topic",
    [
        (FileTypes.L2_SCIENCE, SCIENCE_REPORTING_TOPIC),
        (FileTypes.L1_GUIDE_WINDOW, GUIDE_WINDOW_REPORTING_TOPIC),
    ],
)
def test_report_selects_only_its_topic(monkeypatch, report_type, topic):
    fetch = Mock(side_effect=RuntimeError("stop after parameter selection"))
    monkeypatch.setattr(report, "fetch_parameters_from_path", fetch)

    with pytest.raises(RuntimeError, match="stop after parameter selection"):
        report.report_function(report_type)

    fetch.assert_called_once_with(
        AWS_PARAMETER_PATH, expected_parameters=AWS_DBS + [PARQUET_FILE_BUCKET, topic]
    )