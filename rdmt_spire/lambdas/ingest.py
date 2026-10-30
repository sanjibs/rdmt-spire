import json
import logging
from datetime import datetime, timezone

import boto3
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..constants.codes import StatusCodes
from ..constants.dmd import FileTypes, ReprocessingStates
from ..constants.lambdas import (
    AWS_DBS,
    AWS_MONITOR_QUEUES,
    AWS_PARAMETER_PATH,
    DB_NAME,
    DB_SECRET_NAME,
    ESSENTIAL_L2_MONITOR_QUEUE,
    ESSENTIAL_MONITOR_FROM_EXPOSURE_TYPE,
    GUIDE_WINDOW_MONITOR_QUEUE,
    L2_SCIENCE_EXPOSURE_CONFIG,
    VALID_L2_SCIENCE_EXPOSURE_TYPES,
    MessageKeys,
    MonitorGroupName,
)
from ..db_tables.gw_tables import L1GuideWindowMetaTable
from ..db_tables.sci_tables import L2ScienceMetaTable, L4SourceCatalogL2MetaTable
from ..utilities.aws_utils import fetch_parameters_from_path, get_sqs_url
from ..utilities.db_utils import connect_to_db
from ..utilities.utils import get_info_from_filename

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def ingest_single(message_dict, notification_datetime, aws_account_id):
    """
    Ingest a single SQS-originated message into the metadata store and (optionally)
    enqueue a follow-on monitor task.

    This function:
      1) Parses the incoming message into a metadata table object.
      2) Connects to the database and determines the appropriate reprocessing number
         for the file.
      3) Persists a new metadata row.
      4) Enriches the message with monitor fields and, if applicable, sends it to a
         testing monitor SQS queue.

    Parameters
    ----------
    message_dict : dict
        Message payload describing the file and its context. This is expected to contain
        at least:
          - ``'filename'`` : str
              Name of the file being ingested (used for logging and in the success message).
          - ``'fileType'`` : str
              File type indicator. When equal to ``'science_wfi_level_2'``, a message is
              sent to the testing monitor SQS queue.
        Additional fields are consumed by ``create_table_class_from_message`` and
        ``determine_reprocessing_num`` as needed.
    notification_datetime : datetime.datetime
        Timestamp to be recorded on the metadata row (assigned to
        ``meta_table.dmd_notify_datetime``).
    aws_account_id : str
        AWS account ID used to resolve SQS queue URLs.

    Returns
    -------
    dict
        A response object with the following structure:
        - ``'statusCode'`` : StatusCodes
            ``StatusCodes.SUCCESS`` on success, or
            ``StatusCodes.REPROCESS_NUM_DETERMINATION_FAIL`` if reprocessing number
            determination fails.
        - ``'body'`` : list of dict
            A list containing a single dict with the key ``'error_message'`` holding a
            human-readable message (success or error detail).

    """
    # Unpack the message attributes into a table class
    meta_table = create_table_class_from_message(message_dict)
    meta_table.dmd_notify_datetime = notification_datetime

    # collect relevant parameters from AWS Parameter Store
    params = fetch_parameters_from_path(AWS_PARAMETER_PATH, expected_parameters=AWS_MONITOR_QUEUES + AWS_DBS)

    # Connecting to the database
    logger.info('connecting to the database')
    sql_engine = connect_to_db(database_name=params[DB_NAME], secret_name=params[DB_SECRET_NAME])

    # Check for instances of the filename in the metadata table
    logger.info('getting previous instances of file')
    previous_reprocess_nums_list = get_previous_file_reprocess_numbers(sql_engine, [meta_table])

    # Determine reprocessing number
    logger.info('Determining reprocessing number.')
    reprocess_number, rep_num_err = determine_reprocessing_number(message_dict, previous_reprocess_nums_list[0])

    if rep_num_err is None:
        meta_table.reprocess_number = reprocess_number
    else:
        logger.error(rep_num_err)
        return {'statusCode': StatusCodes.REPROCESS_NUM_DETERMINATION_FAIL,
            'body': [{'error_message': rep_num_err}]}

    # We need to save the exposure type from the metadata table before we write the table and it gets detached
    exposure_type = meta_table.exposure_type

    logger.info('Writing new row to metadata table.')
    write_new_metadata_table_rows(sql_engine, [meta_table])

    logger.info('Successfully wrote new metadata table row.')
    logger.info('Preparing message for monitor queue.')

    message_dict[MessageKeys.FUNCTION_TYPE] = 'monitor'
    message_dict[MessageKeys.REPROCESS_NUMBER] = reprocess_number

    logger.info('Sending message...')
    sqs_client = boto3.client("sqs", region_name='us-east-1')

    if message_dict[MessageKeys.FILE_TYPE] == FileTypes.L2_SCIENCE:
        message_sqs_url = get_sqs_url(
            params[ESSENTIAL_L2_MONITOR_QUEUE],
            account_id=aws_account_id,
            sqs_client=sqs_client
        )
        # Determine which essential monitor to use
        if exposure_type in ESSENTIAL_MONITOR_FROM_EXPOSURE_TYPE:
            essential_monitor_name = ESSENTIAL_MONITOR_FROM_EXPOSURE_TYPE[exposure_type]
        else:
            logger.error(f"Unexpected exposure type: {exposure_type}")
            return {'statusCode': StatusCodes.UNEXPECTED_EXPOSURE_TYPE,
                    'body': [{'error_message': f'Unexpected exposure type for L2 Science File: {exposure_type}'}]}
        logger.info(f'Running {essential_monitor_name} on L2 science file.')
        message_dict[MessageKeys.MONITOR_NAME] = essential_monitor_name

    elif message_dict[MessageKeys.FILE_TYPE] == FileTypes.L1_GUIDE_WINDOW:
        message_sqs_url = get_sqs_url(
            params[GUIDE_WINDOW_MONITOR_QUEUE],
            account_id=aws_account_id,
            sqs_client=sqs_client
        )
        logger.info('Ingesting L1 guide window file.')
        message_dict[MessageKeys.MONITOR_NAME] = MonitorGroupName.ESSENTIAL_GW
    else:
        logger.error(f"Unexpected file type: {message_dict[MessageKeys.FILE_TYPE]}. No monitor message will be sent.")
        return {'statusCode': StatusCodes.UNEXPECTED_FILE_TYPE,
                'body': [{'error_message': f'Unexpected file type: {message_dict[MessageKeys.FILE_TYPE]}. No monitor message will be sent.'}]}

    try:
        response = sqs_client.send_message(
            QueueUrl=message_sqs_url,
            MessageBody=json.dumps(message_dict)
        )
        logger.info(f'Sent message. Response: {response}')
    except Exception as e:
        logger.error(f"Failed in sending message: {e}")
        return {'statusCode': StatusCodes.SQS_SEND_FAIL,
                'body': [{'error_message': f'Failed in sending message: {e}'}]}

    logger.info(f'Successfully ingested {message_dict[MessageKeys.FILENAME]} (reprocess number = {reprocess_number}).')
    return {'statusCode': StatusCodes.SUCCESS,
                'body': [{'error_message': f'Ingested {message_dict[MessageKeys.FILENAME]} (reprocess number = {reprocess_number}) successfully.'}]}

def create_table_class_from_message(message_dict):
    """
    Create and populate a metadata table object from a DMD message.

    This function interprets a message dictionary produced by the DMD
    notification system, constructs the appropriate metadata table class
    (currently only for L2 science and L1 guide window files), and populates its fields using
    both message contents and parsed filename information.

    Parameters
    ----------
    message_dict : dict
        Dictionary containing metadata extracted from the DMD notification.
        Expected keys include:
        - 'fileType' : FileTypes enum
            Indicates the type of file received.
        - 'filename' : str
            The name of the file.
        - 'archiveBucket' : str
            The storage bucket where the file is archived.
        - 'archiveObjectKey' : str
            The key/path of the archived file.
        - 'fileCreationTimestamp' : str
            ISO‑8601 timestamp indicating when the file was created,
            e.g., ``'2024-05-12T18:23:45.123Z'``.

    Returns
    -------
    meta_table : L2ScienceMetaTable or L1GuideWindowMetaTable
        Populated metadata table instance containing:
        - filename
        - archive bucket and key
        - file creation datetime (UTC)
        - program number
        - exposure number
        - visit ID
        - detector
        - optical element

    """
    if message_dict[MessageKeys.FILE_TYPE] == FileTypes.L2_SCIENCE:
        # Confirm the exposure type is expected values
        exposure_type = message_dict[MessageKeys.EXPOSURE_TYPE]
        if exposure_type not in VALID_L2_SCIENCE_EXPOSURE_TYPES:
            raise ValueError(f"Unexpected exposure type in L2 science file message: {exposure_type}")
        meta_table = L2ScienceMetaTable()
        # Save metadata from DMD notification
        meta_table.filename = message_dict[MessageKeys.FILENAME]
        meta_table.archive_bucket = message_dict[MessageKeys.ARCHIVE_BUCKET]
        meta_table.archive_key = message_dict[MessageKeys.ARCHIVE_OBJECT_KEY]
        meta_table.file_created_datetime = datetime.strptime(message_dict[MessageKeys.FILE_CREATION_TIMESTAMP], '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
        exposure_config = L2_SCIENCE_EXPOSURE_CONFIG[exposure_type]
        meta_table.exposure_type = exposure_config["exposure_type"]
        for status_column, status_value in exposure_config["status_values"].items():
            setattr(meta_table, status_column, int(status_value))

        # Save metadata from the filename
        obs_info = get_info_from_filename(message_dict[MessageKeys.FILENAME], FileTypes.L2_SCIENCE)
        meta_table.program_number = obs_info['program_num']
        meta_table.execution_number = obs_info['execution_num']
        meta_table.pass_number = obs_info['pass_num']
        meta_table.segment_number = obs_info['segment_num']
        meta_table.observation_number = obs_info['obs_num']
        meta_table.visit_number = obs_info['visit_num']
        meta_table.exposure_number = obs_info['exposure_num']
        meta_table.visit_id = obs_info['visit_id']
        meta_table.detector = obs_info['detector']
        meta_table.optical_element = obs_info['optical_element']

    elif message_dict[MessageKeys.FILE_TYPE] == FileTypes.L1_GUIDE_WINDOW:
        meta_table = L1GuideWindowMetaTable()
        # Save metadata from DMD notification
        meta_table.filename = message_dict[MessageKeys.FILENAME]
        meta_table.archive_bucket = message_dict[MessageKeys.ARCHIVE_BUCKET]
        meta_table.archive_key = message_dict[MessageKeys.ARCHIVE_OBJECT_KEY]
        meta_table.file_created_datetime = datetime.strptime(message_dict[MessageKeys.FILE_CREATION_TIMESTAMP], '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)

        # Save metadata from the filename
        obs_info = get_info_from_filename(message_dict[MessageKeys.FILENAME], FileTypes.L1_GUIDE_WINDOW)
        meta_table.program_number = obs_info['program_num']
        meta_table.gw_acquisition_number = obs_info['gw_acquisition_num']
        meta_table.acquisition_id = f"{obs_info['visit_id']}_{obs_info['gw_acquisition_num']}"
        meta_table.visit_id = obs_info['visit_id']
        meta_table.detector = obs_info['detector']
        meta_table.optical_element = obs_info['optical_element']
    
    elif message_dict[MessageKeys.FILE_TYPE] == FileTypes.L4_SOURCE_CATALOG_L2:
        meta_table = L4SourceCatalogL2MetaTable()
        # Save metadata from DMD notification
        meta_table.filename = message_dict[MessageKeys.FILENAME]
        meta_table.filename_l2 = message_dict[MessageKeys.FILENAME].removesuffix("_cat.parquet")+'_cal.asdf'
        meta_table.archive_bucket = message_dict[MessageKeys.ARCHIVE_BUCKET]
        meta_table.archive_key = message_dict[MessageKeys.ARCHIVE_OBJECT_KEY]


    return meta_table

def get_previous_file_reprocess_numbers(engine, meta_tables):
    """
    Retrieve historical reprocessing numbers for a collection of metadata table instances.

    This function groups the provided metadata table objects by their SQLAlchemy table class,
    queries the database for all previous rows associated with each file (based on filename),
    and returns a list of lists containing the historical reprocess number values for each
    corresponding metadata table in the input list.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine used to create database sessions and execute queries.
    meta_tables : list of Base
        A list of SQLAlchemy ORM-mapped table instances. Each instance must have
        `filename` and `reprocess_number` attributes.

    Returns
    -------
    list of list of int
        A list where each element corresponds to one input metadata table. Each element
        is a list of all reprocess number values retrieved from the database for rows
        sharing the same filename as that metadata table.

    """
    # identify which table classes need to be queried
    table_classes = [mt.__class__ for mt in meta_tables]
    unique_classes = list(set(table_classes))

    # generate the SQLalchemy statements for the queries to get all previous iterations of the files
    stmts = []
    for unique_table_class in unique_classes:
        filenames = [mt.filename for mt in meta_tables if isinstance(mt, unique_table_class)]
        stmts.append(select(unique_table_class).where(unique_table_class.filename.in_(filenames)))

    # execute the statements and save the resulting rows in a list
    db_rows = []
    with Session(engine) as session:
        for stmt in stmts:
            result_rows = session.scalars(stmt).all()
            db_rows += result_rows

    # generate a list of reprocessing numbers for each input metadata table
    reprocess_nums_list = []
    for mt in meta_tables:
        reprocess_nums_list.append([row.reprocess_number for row in db_rows if row.filename == mt.filename])

    return reprocess_nums_list

def determine_reprocessing_number(message_dict, previous_reprocess_nums):
    """
    Determine the next reprocessing number and any consistency error.

    This function inspects the incoming message's ``reprocessingState`` and the
    history of previously used reprocessing numbers to compute the appropriate
    next ``reprocess_num`` or return an error message describing an
    inconsistency.

    Parameters
    ----------
    message_dict : dict
        A dictionary that must contain the key ``'reprocessingState'`` whose value
        is a member of ``ReprocessingStates`` (e.g., ``ReprocessingStates.PROMPT``,
        ``ReprocessingStates.REPROCESSED``, ``ReprocessingStates.DATA_RELEASE``).
    previous_reprocess_nums : Sequence[int]
        A (possibly empty) sequence of previously assigned reprocessing numbers
        for the file. Typically a list of integers.

    Returns
    -------
    reprocess_num : int
        The computed reprocessing number. Defaults to ``0`` for prompt/initial
        state or when no increment applies. If an inconsistency is detected,
        this will be the best-effort computed value (often ``0``) and
        the error will be described in ``rep_num_err``.
    rep_num_err : Optional[str]
        ``None`` if no inconsistency is found; otherwise, a human-readable error
        message describing the mismatch between the message state and the
        metadata (e.g., file not in metadata but state indicates a reprocessed/release).

    """
    rep_num_err = None
    reprocess_num = 0
    if message_dict[MessageKeys.REPROCESSING_STATE] == ReprocessingStates.PROMPT:
        if len(previous_reprocess_nums) > 0:
            rep_num_err = f'File is already in metadata table, but notification has reprocessingState={ReprocessingStates.PROMPT}.'
    elif message_dict[MessageKeys.REPROCESSING_STATE] == ReprocessingStates.REPROCESSED:
        if len(previous_reprocess_nums) > 0:
            reprocess_num = 1 + max(previous_reprocess_nums)
        else:
            rep_num_err = f'File not in metadata table, but notification has reprocessingState={ReprocessingStates.REPROCESSED}.'
    elif message_dict[MessageKeys.REPROCESSING_STATE] == ReprocessingStates.DATA_RELEASE:
        if len(previous_reprocess_nums) > 0:
            reprocess_num = 1 + max(previous_reprocess_nums)
        else:
            rep_num_err = f'File not in metadata table, but notification has reprocessingState={ReprocessingStates.DATA_RELEASE}.'

    return reprocess_num, rep_num_err

def write_new_metadata_table_rows(engine, rows_to_add):
    """
    Add new ORM rows to the metadata table and commit the transaction.

    Opens a short-lived SQLAlchemy session bound to the provided engine,
    adds each ORM-mapped instance from ``rows_to_add`` to the session, and
    commits the transaction. This function does not return a value; it
    persists the provided instances to the database.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine or sqlalchemy.engine.base.Engine
        A SQLAlchemy Engine bound to the target database.
    rows_to_add : Iterable[sqlalchemy.orm.DeclarativeBase] or Iterable[Any]
        An iterable of SQLAlchemy ORM-mapped instances (e.g., declarative
        model objects) to be inserted. Each element should be a transient or
        pending instance appropriate for insertion.

    Returns
    -------
    None
        This function has no return value. On success, the rows are written
        to the database.

    """
    with Session(engine) as session:
        for rc in rows_to_add:
            session.add(rc)
        session.commit()

#### TODO: NOT YET IMPLEMENTED. Error handling when processing batched messages needs to be better understood.
def ingest_batch(message_dicts, notification_datetimes):
    """
    A still to be implemented version of ingest_single that handles a larger batch size of messages.

    """
    # Unpack the message attributes into a table class
    meta_tables = [create_table_class_from_message(md) for md in message_dicts]
    for i, mt in enumerate(meta_tables):
        mt.dmd_notify_datetime = notification_datetimes[i]

    logger.info(f'dmd notify times: {[mt.dmd_notify_datetime for mt in meta_tables]}')

    # Connecting to the database
    # collect relevant parameters from AWS Parameter Store
    params = fetch_parameters_from_path(AWS_PARAMETER_PATH, expected_parameters=AWS_MONITOR_QUEUES + AWS_DBS)

    # Connecting to the database
    logger.info('connecting to the database')
    sql_engine = connect_to_db(database_name=params[DB_NAME], secret_name=params[DB_SECRET_NAME])

    # Check for instances of the filenames in the metadata tables
    logger.info('Retrieving previous instances of the files')
    previous_reprocess_num_list = get_previous_file_reprocess_numbers(sql_engine, meta_tables)

    logger.info(f'reprocess_nums_list: {previous_reprocess_num_list}')
    #logger.info('Determining reprocessing number')
    #for message_dict, meta_table, previous_reprocess_nums in zip(message_dicts, meta_tables,#previous_reprocess_num_list):
    #    reprocess_num, rep_num_err = determine_reprocessing_num(message_dict, previous_reprocess_nums)
    #    if rep_num_err is None:
    #        meta_table.reprocess_number = reprocess_num
    #    else:
    #        logger.error(rep_num_err)
    #        return {'statusCode': StatusCodes.REPROCESS_NUM_DETERMINATION_FAIL,
    #            'body': [{'error_message': rep_num_err}]}

    return {'statusCode': StatusCodes.NOT_YET_IMPLEMENTED,
            'body': [{'error_message':'ingest_batch() is not yet implemented.'}]}
