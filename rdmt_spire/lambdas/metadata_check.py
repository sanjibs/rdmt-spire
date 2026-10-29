import json
import logging
from datetime import datetime, timedelta

import boto3
from sqlalchemy import and_, case, or_, select, tuple_, update
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import Session

from ..constants.codes import StatusCodes
from ..constants.database import MonitorStatus
from ..constants.lambdas import (
    ASTROMETRY_MONITOR_QUEUE,
    AWS_DBS,
    AWS_MONITOR_QUEUES,
    AWS_PARAMETER_PATH,
    DB_NAME,
    DB_SECRET_NAME,
    FunctionTypes,
    MessageKeys,
)
from ..db_tables.sci_tables import L2ScienceMetaTable
from ..utilities.aws_utils import fetch_parameters_from_path, get_sqs_url
from ..utilities.db_utils import connect_to_db

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def metadata_check_function(message_dict, aws_account_id):
    """
    Evaluate pending metadata checks and enqueue monitor messages.

    The function scans ``L2ScienceMetaTable`` for rows with status values of
    ``-2`` (pending), applies per-monitor selection rules, updates status
    columns to ``0`` (selected) or ``-1`` (not selected), and sends one SQS
    message per selected astrometry row.

    Parameters
    ----------
    message_dict : dict
        Lambda payload. Currently unused, but retained for handler interface
        compatibility.
    aws_account_id : str
        AWS account ID used to resolve SQS queue URLs.

    Returns
    -------
    dict
        API-style response containing ``statusCode`` and a ``body`` message.

    """

    params = fetch_parameters_from_path(AWS_PARAMETER_PATH, expected_parameters=AWS_MONITOR_QUEUES + AWS_DBS)

    all_status_columns = [
        column.key for column in sa_inspect(L2ScienceMetaTable).columns if column.key.endswith("_status")
    ]

    metadata_check_status_columns = [
        "astrometry_status",
    ]

    logger.info(f"Running Metadata check for columns: {', '.join(metadata_check_status_columns)}")

    logger.info('Connecting to the database...')
    sql_engine = connect_to_db(database_name=params[DB_NAME], secret_name=params[DB_SECRET_NAME])

    with Session(sql_engine) as session:
        # Fetch all rows that have any pending monitor status (-2).
        logger.info("Querying database for new rows to check ...")
        # Build a filter condition that checks for any status column being -2.
        all_min2_statuses = or_(
            *[getattr(L2ScienceMetaTable, col) == MonitorStatus.PENDING_CHECK for col in all_status_columns]
        )
        # Select only the primary key columns and the status columns for efficiency.
        ids_and_status_columns = [
            L2ScienceMetaTable.filename,
            L2ScienceMetaTable.reprocess_number,
            L2ScienceMetaTable.exp_start_datetime,
            *[getattr(L2ScienceMetaTable, col) for col in all_status_columns],
        ]
        # Query the database for rows that have any status column equal to -2.
        pk_rows = session.execute(
            select(*ids_and_status_columns).where(all_min2_statuses)
            .order_by(L2ScienceMetaTable.exp_start_datetime.asc())
        ).all()

        # Check for any rows that have -2 in non-supported status columns. If any are found, raise an error.
        non_metadata_check_status_columns = [
            col for col in all_status_columns if col not in metadata_check_status_columns
        ]
        if non_metadata_check_status_columns:
            bad_columns = []
            for row in pk_rows:
                for col in non_metadata_check_status_columns:
                    if getattr(row, col) == MonitorStatus.PENDING_CHECK:
                        logger.error(f"Row with filename={row.filename}, reprocess_number={row.reprocess_number} has {col} == -2, which is not supported for metadata checks.")
                        bad_columns.append(col)
            if len(bad_columns) > 0:
                raise ValueError(
                    "Found rows with -2 in unsupported status columns: "
                    f"{', '.join(set(bad_columns))}"
                )

        # If no rows are found, return a success message indicating that no messages will be sent to SQS.
        if not pk_rows:
            logger.info("No rows found with any *_status == -2. No messages will be sent to SQS.")
            return {
                "statusCode": StatusCodes.SUCCESS,
                "body": [{"message": "No rows found with any *_status == -2. No messages sent to SQS."}],
            }
        logger.info(f"Found {len(pk_rows)} rows with any *_status == -2.")

        logger.info("Defining selection rules for metadata checks ...")
        # Define selection rules for each monitor. Each rule is a list of tuples where each tuple contains a condition and the corresponding status value to set. The conditions are SQLAlchemy expressions that evaluate to True or False for each row. The status values are integers that indicate whether the monitor should run (0) or not (-1).
        rules = {}
        logger.info("Defining selection rule for astrometry ...")
        astrometry_rule = every_other_rule(pk_rows, 'astrometry_status')
        # TODO: test once we have more data spread across time
        # astrometry_rule = time_based_rule(session, pk_rows, 'astrometry_status')
        rules['astrometry_status'] = astrometry_rule

        # Validate selection rules and build SQL CASE statements.
        logger.info("Validating selection rules and building sql statement ...")
        rule_statements = {}
        for col, rule in rules.items():
            if isinstance(rule, list) and all(isinstance(r, tuple) and len(r) == 2 for r in rule):
                rule_statements[col] = case(*rule, else_=int(MonitorStatus.NOT_APPLICABLE))
            else:
                raise ValueError(f"Invalid rule format for {col}: {rule}. Must be a tuple or list of tuples.")
            
        # Apply rule-based status updates to the pending rows.
        logger.info("Updating rows in the database based on selection rules ...")
        
        # isolate the primary key values for the rows to update
        pk_values = [(row.filename, row.reprocess_number) for row in pk_rows]

        # Build the update statement to set the status columns based on the selection rules. The update will only affect rows that match the primary key values of the pending rows.
        stmt = (
            update(L2ScienceMetaTable)
            .where(
                tuple_(
                    L2ScienceMetaTable.filename,
                    L2ScienceMetaTable.reprocess_number,
                ).in_(pk_values)
            )
            .values(rule_statements)
            .execution_options(synchronize_session=False)
        )

        session.execute(stmt)
        session.commit()

        # Reload rows selected for at least one metadata check.
        logger.info("Querying database for rows that have been updated to *_status == 0 ...")

        # Fetch all rows that were just updated and have any of the metadata check status columns equal to 0 (selected).
        updated_rows = session.execute(
            select(L2ScienceMetaTable).where(
                and_(
                    tuple_(
                        L2ScienceMetaTable.filename,
                        L2ScienceMetaTable.reprocess_number,
                    ).in_(pk_values),
                    or_(*[getattr(L2ScienceMetaTable, col) == MonitorStatus.NEEDS_TO_RUN for col in metadata_check_status_columns])
                )
            )
        ).scalars().all()

    # if no rows were selected for any metadata check, return a success message indicating that no messages will be sent to SQS.
    if not updated_rows:
        return {
            "statusCode": StatusCodes.SUCCESS,
            "body": [{"message": "No rows found with any *_status == 0 after update. No messages sent to SQS."}],
        }
    
    logger.info(f"Found {len(updated_rows)} rows with *_status == 0 after update. Sending messages to SQS.")
    sqs = boto3.client("sqs", region_name='us-east-1')

    # gather all rows that have the various _status columns == 0 and send them to the appropriate SQS queue for processing by the monitor lambdas.
    astrometry_rows = [row for row in updated_rows if row.astrometry_status == MonitorStatus.NEEDS_TO_RUN]
    if astrometry_rows:
        astrometry_queue = get_sqs_url(
            params[ASTROMETRY_MONITOR_QUEUE], 
            account_id=aws_account_id
        )
        for row in astrometry_rows:
            monitor_dict = generate_message_dict_from_metadata_table(row, 'astrometry')
            try:
                response = sqs.send_message(
                    QueueUrl=astrometry_queue,
                    MessageBody=json.dumps(monitor_dict)
                )
                logger.info(f'Message sent for astrometry. Response: {response}')
            except Exception as e:
                logger.error(f"Failed in sending message: {e}")
                # TODO: how do we want to handle this error correctly?
                raise e

    logger.info('Successfully sent all messages. Metadata check complete,')

    return {'statusCode': StatusCodes.SUCCESS,
                'body': [{'message':'metadata_check_function ran successfully.'}]} 

def every_other_rule(pk_rows, status_column):
    """
    Select every other pending row for the given monitor.

    Parameters
    ----------
    pk_rows : list of table rows (named tuples)
        Rows that contain at least ``filename``, ``reprocess_number``, and
        the column named by ``status_column``.
    status_column : str
        Name of the ``*_status`` column that this rule targets.

    Returns
    -------
    list[tuple]
        Rule tuples consumable by ``sqlalchemy.case`` in the form
        ``[(condition, status_value)]``.
    """
    indices = [
        (row.filename, row.reprocess_number)
        for row in pk_rows
        if getattr(row, status_column) == MonitorStatus.PENDING_CHECK
    ]
    selected_pks = []
    for i, (filename, reprocess_number) in enumerate(indices):
        if i % 2 == 0:  # Select every other row (even index)
            selected_pks.append((filename, reprocess_number))

    should_run = tuple_(
        L2ScienceMetaTable.filename,
        L2ScienceMetaTable.reprocess_number,
    ).in_(selected_pks)

    return [(should_run, int(MonitorStatus.NEEDS_TO_RUN))]
    
def time_based_rule(session, pk_rows, status_column):
    """
    Select rows spaced by at least one hour from the last selected run time.
    
    Parameters
    ----------
    session : sqlalchemy.orm.Session
        An active SQLAlchemy session used to query the database.
    pk_rows : list of tuples
        Rows that contain at least ``filename``, ``reprocess_number``, and
        ``exp_start_datetime``.
    status_column : str
        Name of the ``*_status`` column that this rule targets.

    Returns
    -------
    list[tuple]
        Rule tuples consumable by ``sqlalchemy.case`` in the form
        ``[(condition, status_value)]``.
    """
    astrometry_time_delta = timedelta(hours=1)
    # Limit look back for prior activity when computing cadence.
    query_time_delta = timedelta(days=30)
    status_col = getattr(L2ScienceMetaTable, status_column)
    last_astrometry_time = session.execute(
        select(L2ScienceMetaTable.exp_start_datetime)
        .where(and_(
            status_col.in_([MonitorStatus.NEEDS_TO_RUN, MonitorStatus.COMPLETE]),
            L2ScienceMetaTable.exp_start_datetime.is_not(None),
            L2ScienceMetaTable.exp_start_datetime > (datetime.now() - query_time_delta)
        ))
        .order_by(L2ScienceMetaTable.exp_start_datetime.desc())
        .limit(1)
    ).scalar()

    selected_pks = []
    last_time = last_astrometry_time
    indices = [
        (row.filename, row.reprocess_number, row.exp_start_datetime)
        for row in pk_rows
        if getattr(row, status_column) == MonitorStatus.PENDING_CHECK
    ]
    for filename, reprocess_number, exp_start_dt in indices:
        if last_time is None or exp_start_dt > last_time + astrometry_time_delta:
            selected_pks.append((filename, reprocess_number))
            last_time = exp_start_dt

    should_run = tuple_(
        L2ScienceMetaTable.filename,
        L2ScienceMetaTable.reprocess_number,
    ).in_(selected_pks)

    return [(should_run, int(MonitorStatus.NEEDS_TO_RUN))]

def generate_message_dict_from_metadata_table(metadata_table_class, monitor_name):
    """
    Build a monitor payload from a metadata ORM row.

    Parameters
    ----------
    metadata_table_class : SQLAlchemy table class
        ORM row object compatible with ``L2ScienceMetaTable`` fields.
    monitor_name : str
        The name of the monitor associated with this message.

    Returns
    -------
    dict
        Message payload consumed by monitor lambdas.

    """
    monitor_dict = {
        MessageKeys.ARCHIVE_BUCKET: metadata_table_class.archive_bucket,
        MessageKeys.ARCHIVE_OBJECT_KEY: metadata_table_class.archive_key,
        MessageKeys.FILE_TYPE: metadata_table_class.file_type,
        MessageKeys.FILENAME: metadata_table_class.filename,
        MessageKeys.FUNCTION_TYPE: FunctionTypes.MONITOR,
        MessageKeys.MONITOR_NAME: monitor_name,
        MessageKeys.REPROCESS_NUMBER: metadata_table_class.reprocess_number,
    }

    return monitor_dict