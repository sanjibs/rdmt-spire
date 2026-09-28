import copy
import logging
from datetime import timezone
from pathlib import Path

import asdf
from sqlalchemy.orm import Session

from ..constants.codes import StatusCodes
from ..constants.dmd import FileTypes
from ..constants.lambdas import (
    AWS_DBS,
    AWS_PARAMETER_PATH,
    AWS_S3_BUCKETS,
    DB_NAME,
    DB_SECRET_NAME,
    RDMT_SPIRE_L4_BUCKET,
    RDMT_SPIRE_RDATA_BUCKET,
    MessageKeys,
)
from ..db_tables.gw_tables import L1GuideWindowMetaTable, L1GuideWindowResultsTable
from ..db_tables.sci_tables import L2ScienceMetaTable, L2ScienceResultsTable
from ..manager import MonitorManager
from ..utilities.aws_utils import fetch_parameters_from_path, load_s3_object
from ..utilities.db_utils import connect_to_db

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def monitor_function(message_dict):
    """
    Orchestrate running one or more monitors on an ASDF file from S3 and archive results.

    This function:
    1. Loads an ASDF product from Amazon S3 using the bucket and object key from
       ``message_dict``.
    2. Instantiates a :class:`MonitorManager` with the opened ASDF and the target
       monitor name(s), then executes the monitor pipeline via ``process()``.
    3. Extracts metadata from the ASDF, connects to the database, archives monitor
       metrics into the appropriate results table, and populates any missing metadata in
       the metadata table.
    4. Commits the transaction and returns a response suitable for an AWS Lambda
       handler (via ``MonitorManager.response_for_lambda()``).
    
    If loading the ASDF content fails, it returns a structured failure response without
    attempting any database operations.

    Parameters
    ----------
    message_dict : dict
        Execution parameters and routing information from the triggering SQS message.

    Returns
    -------
    dict
        A response payload intended for AWS Lambda handlers.

    """
    s3_bucket = message_dict[MessageKeys.ARCHIVE_BUCKET]
    s3_object_key = message_dict[MessageKeys.ARCHIVE_OBJECT_KEY]
    try:
        content = load_s3_object(s3_bucket, s3_object_key)
        af = asdf.open(content)
        logger.info(f'Loaded {message_dict[MessageKeys.FILENAME]}.')
    except Exception as e:
        logger.error(f'error loading the asdf file: {e}')
        return {'statusCode': StatusCodes.FAILURE,
                'body': [{'error_message': e}]}
    
    params = fetch_parameters_from_path(AWS_PARAMETER_PATH, expected_parameters=AWS_DBS+AWS_S3_BUCKETS)

    monitor_config = generate_monitor_config(message_dict, params)

    logger.info(f'Running {message_dict[MessageKeys.MONITOR_NAME]} monitor(s).')
    monitor_manager = MonitorManager(af, message_dict[MessageKeys.MONITOR_NAME], monitor_config=monitor_config)
    monitor_manager.process()
    logger.info('Finished monitor execution.')

    # clean up memory
    metadata_dict = copy.deepcopy(af['roman']['meta'])
    af.close()
    del content

    # determine the database tables to use
    if message_dict[MessageKeys.FILE_TYPE] == FileTypes.L2_SCIENCE:
        metadata_table_class = L2ScienceMetaTable
        results_table_class = L2ScienceResultsTable
    elif message_dict[MessageKeys.FILE_TYPE] == FileTypes.L1_GUIDE_WINDOW:
        metadata_table_class = L1GuideWindowMetaTable
        results_table_class = L1GuideWindowResultsTable

    # Connecting to the database
    logger.info('Connecting to the database.')
    sql_engine = connect_to_db(database_name=params[DB_NAME], secret_name=params[DB_SECRET_NAME])

    with Session(sql_engine) as session:
        # Update the results table monitor results
        logger.info('Archiving monitor metrics.')
        monitor_manager.archive(
            session, 
            message_dict[MessageKeys.FILENAME], 
            message_dict[MessageKeys.REPROCESS_NUMBER],
            results_table_class
        )
        
        # Check for missing metadata in table and populate it
        logger.info('Checking metadata from file.')
        update_metadata_table(
            session, 
            message_dict[MessageKeys.FILENAME], 
            message_dict[MessageKeys.REPROCESS_NUMBER], 
            message_dict[MessageKeys.MONITOR_NAME], 
            metadata_table_class,
            metadata_dict
        )

        session.commit()

    logger.info(f'Successfully ran {message_dict[MessageKeys.MONITOR_NAME]} monitor(s) on {message_dict[MessageKeys.FILENAME]} (reprocess number = {message_dict[MessageKeys.REPROCESS_NUMBER]}) and stored results in database.')

    return monitor_manager.response_for_lambda()


def generate_monitor_config(message_dict, params):
    """
    Generate a monitor configuration dictionary based on the message and AWS parameters.

    This function constructs a configuration dictionary for monitors, particularly for
    the astrometry monitor, by extracting necessary parameters from the provided
    ``message_dict`` and AWS parameter store values.

    Parameters
    ----------
    message_dict : dict
        The input message containing execution parameters, including the monitor name.
    params : dict
        A dictionary of parameters fetched from AWS Parameter Store, expected to include
        paths for S3 buckets and other necessary configuration values.

    Returns
    -------
    dict
        A configuration dictionary suitable for initializing monitors. For example,
        it may contain the data directory path for the astrometry monitor.

    """
    monitor_config = {}
    
    # if message_dict[MessageKeys.MONITOR_NAME] == "astrometry":
    #     monitor_config["astrometry"] = {
    #         "datadir": params[ASTROMETRY_MONITOR_DATA_BUCKET]
    #     }
    if message_dict[MessageKeys.MONITOR_NAME] in ["source_catalog","photoastro","astrometry"]:
        config={}
        config["RDMT_SPIRE_LDATA_DIR"] = str(Path(__file__).resolve().parent.parent / "rdmt_data")
        config["RDMT_SPIRE_L2_DIR"] = ""
        config["RDMT_SPIRE_L4_DIR"] = f"s3://{params[RDMT_SPIRE_L4_BUCKET]}"
        config["RDMT_SPIRE_RDATA_DIR"] = f"s3://{params[RDMT_SPIRE_RDATA_BUCKET]}"
        monitor_config[message_dict[MessageKeys.MONITOR_NAME]]=config
    
    return monitor_config


def update_metadata_table(session, filename, reprocess_number, monitor_name, metadata_table_class, metadata_dict):
    """
    Update a row in the metadata table for a given file/reprocess and mark a monitor as successful.

    This function retrieves (or assumes the existence of) a row identified by the composite
    key ``(filename, reprocess_number)`` from the SQLAlchemy-backed metadata table and:
    1. Sets the monitor-specific status column (``<monitor_name>_status``) to ``1`` if that
       column exists; otherwise logs a warning without failing.
    2. If key observation-related fields have not yet been populated on the row
       (determined by a falsy ``observation_id``), initializes them from ``metadata_dict``:
       For L2 science files:
            - ``observation_id``
            - ``exp_start_datetime`` (converted to timezone-aware UTC)
            - ``romancal_version``
            - ``crds_context``
            - ``sdf_version``
       For L1 guide window files:
            - ``acq_start_datetime`` (converted to timezone-aware UTC)
            - ``sdf_version``
    3. Flushes the session so changes are persisted to the current transaction.

    Parameters
    ----------
    session : sqlalchemy.orm.Session
        An active SQLAlchemy session used to load and update the metadata row. The function
        calls ``session.get(...)`` and ``session.flush()``.
    filename : str
        The file identifier (e.g., product or exposure filename) that is part of the
        composite primary key for the metadata table.
    reprocess_number : int
        The reprocessing iteration or version number that, together with ``filename``,
        uniquely identifies the metadata row.
    monitor_name : str
        Logical name of the monitor whose completion status should be recorded. The function
        will look for a column named ``f"{monitor_name}_status"`` on the table.
    metadata_table_class : DeclarativeMeta
        The SQLAlchemy declarative model class representing the metadata table. It must have
        a composite primary key including ``filename`` and ``reprocess_number`` and attributes
        corresponding to the columns updated here (e.g., ``observation_id``,
        ``exp_start_datetime``, ``romancal_version``, ``crds_context``, ``sdf_version``).
    metadata_dict : dict
        A nested dictionary containing source metadata used to initialize the row when
        missing. Should be extracted from the ASDF file that is analyzed 
        (e.g., asdf_file['roman']['meta'])

    Returns
    -------
    None
        The function updates the ORM object in-place and flushes the changes to the current
        transaction.

    """
    meta_row = session.get(metadata_table_class, (filename, reprocess_number))
    col_names = metadata_table_class.__table__.c.keys()

    monitor_status_str = f'{monitor_name}_status'
    if monitor_status_str in col_names:
        setattr(meta_row, monitor_status_str, 1)
    else:
        logger.warning(f"Monitor ({monitor_name}) ran successfully, but {monitor_status_str} is not a column in the metadata table. Not updating status.")
    
    if not meta_row.sdf_version:
        logger.info("Metadata row is missing key fields; populating from ASDF metadata.")
        if metadata_table_class.file_type == FileTypes.L2_SCIENCE:
            meta_row.observation_id = metadata_dict['observation']['observation_id']
            meta_row.exp_start_datetime = metadata_dict['exposure']['start_time'].to_datetime(timezone=timezone.utc)
            meta_row.romancal_version = metadata_dict['calibration_software_version']
            meta_row.crds_context = metadata_dict['ref_file']['crds']['context']
            meta_row.sdf_version = metadata_dict['sdf_software_version']
        elif metadata_table_class.file_type == FileTypes.L1_GUIDE_WINDOW:
            meta_row.acq_start_datetime = metadata_dict['t_start'].to_datetime(timezone=timezone.utc)
            meta_row.sdf_version = metadata_dict['sdf_software_version']
        else:
            logger.warning(f"Metadata table class {metadata_table_class.__name__} has unrecognized file type {metadata_table_class.file_type}. Not populating metadata fields.")

    session.flush()