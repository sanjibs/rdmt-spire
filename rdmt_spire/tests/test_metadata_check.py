import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from ..constants.codes import StatusCodes
from ..constants.database import MonitorStatus
from ..constants.dmd import FileTypes
from ..constants.lambdas import (
    ASTROMETRY_MONITOR_QUEUE,
    DB_NAME,
    DB_SECRET_NAME,
    SOURCE_CATALOG_MONITOR_QUEUE,
    FunctionTypes,
    MessageKeys,
)
from ..db_tables.sci_tables import L2ScienceMetaTable, L4SourceCatalogL2MetaTable
from ..lambdas import metadata_check


def test_metadata_check_matches_l4_metadata_by_l2_filename(monkeypatch):
    """Selected L2 rows must find their L4 metadata using filename_l2."""
    science_row = L2ScienceMetaTable(
        filename="r0034201001001001001_0001_wfi01_f087_cal.asdf",
        reprocess_number=2,
        archive_bucket="science-bucket",
        archive_key="l2/science_cal.asdf",
        essential_status=MonitorStatus.COMPLETE,
        astrometry_status=MonitorStatus.NOT_APPLICABLE,
        source_catalog_status=MonitorStatus.NEEDS_TO_RUN,
    )
    catalog_row = L4SourceCatalogL2MetaTable(
        filename="r0034201001001001001_0001_wfi01_f087_cat.parquet",
        filename_l2=science_row.filename,
        reprocess_number=science_row.reprocess_number,
        archive_bucket="catalog-bucket",
        archive_key="l4/source_cat.parquet",
    )
    pending_row = SimpleNamespace(
        filename=science_row.filename,
        reprocess_number=science_row.reprocess_number,
        essential_status=MonitorStatus.COMPLETE,
        astrometry_status=MonitorStatus.NOT_APPLICABLE,
        source_catalog_status=MonitorStatus.PENDING_CHECK,
    )
    pending_result = Mock()
    pending_result.all.return_value = [pending_row]
    science_result = Mock()
    science_result.scalars.return_value.all.return_value = [science_row]
    catalog_result = Mock()
    catalog_result.scalars.return_value.all.return_value = [catalog_row]
    session = MagicMock()
    session.execute.side_effect = [
        pending_result,
        None,
        science_result,
        catalog_result,
    ]
    session_factory = MagicMock()
    session_factory.return_value.__enter__.return_value = session
    monkeypatch.setattr(metadata_check, "Session", session_factory)
    monkeypatch.setattr(metadata_check, "connect_to_db", Mock())
    monkeypatch.setattr(
        metadata_check,
        "fetch_parameters_from_path",
        Mock(return_value={
            DB_NAME: "test-db",
            DB_SECRET_NAME: "test-secret",
            ASTROMETRY_MONITOR_QUEUE: "astrometry-queue",
            SOURCE_CATALOG_MONITOR_QUEUE: "source-catalog-queue",
        }),
    )
    monkeypatch.setattr(
        metadata_check, "get_sqs_url", Mock(return_value="https://test/queue")
    )
    sqs = Mock()
    monkeypatch.setattr(metadata_check.boto3, "client", Mock(return_value=sqs))

    response = metadata_check.metadata_check_function({}, "123456789012")

    assert response["statusCode"] == StatusCodes.SUCCESS
    sqs.send_message.assert_called_once()
    assert sqs.send_message.call_args.kwargs["QueueUrl"] == "https://test/queue"
    payload = json.loads(sqs.send_message.call_args.kwargs["MessageBody"])
    assert payload == {
        MessageKeys.ARCHIVE_BUCKET: science_row.archive_bucket,
        MessageKeys.ARCHIVE_OBJECT_KEY: science_row.archive_key,
        MessageKeys.FILE_TYPE: FileTypes.L2_SCIENCE,
        MessageKeys.FILENAME: science_row.filename,
        MessageKeys.FUNCTION_TYPE: FunctionTypes.MONITOR,
        MessageKeys.MONITOR_NAME: "source_catalog",
        MessageKeys.REPROCESS_NUMBER: science_row.reprocess_number,
        MessageKeys.L4_ARCHIVE_BUCKET: catalog_row.archive_bucket,
        MessageKeys.L4_ARCHIVE_OBJECT_KEY: catalog_row.archive_key,
    }
