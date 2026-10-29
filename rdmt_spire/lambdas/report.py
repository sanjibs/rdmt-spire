import logging
import os
from dataclasses import dataclass
from datetime import datetime

import boto3
import duckdb

from ..constants.codes import StatusCodes
from ..constants.database import MonitorStatus
from ..constants.dmd import FileTypes
from ..constants.lambdas import (
    AWS_DBS,
    AWS_PARAMETER_PATH,
    AWS_S3_BUCKETS,
    AWS_SNS_TOPICS,
    DB_NAME,
    DB_SECRET_NAME,
    GUIDE_WINDOW_REPORTING_TOPIC,
    PARQUET_FILE_BUCKET,
    SCIENCE_REPORTING_TOPIC,
)
from ..db_tables.gw_tables import L1GuideWindowMetaTable, L1GuideWindowResultsTable
from ..db_tables.sci_tables import L2ScienceMetaTable, L2ScienceResultsTable
from ..utilities.aws_utils import fetch_parameters_from_path
from ..utilities.db_utils import connect_to_db

logger = logging.getLogger()
logger.setLevel(logging.INFO)

tab_str = "    "


@dataclass(frozen=True)
class ReportSpec:
    """Configuration for report-type-specific table and column behavior."""

    dataset_prefix: str
    meta_table_class: type
    parquet_file_prefix: str
    reporting_topic_name: str
    results_table_class: type
    start_time_column: str
    summary_id_column: str

    @property
    def meta_table_name(self) -> str:
        return self.meta_table_class.__tablename__

    @property
    def results_table_name(self) -> str:
        return self.results_table_class.__tablename__


def _get_report_spec(report_type: str = FileTypes.L2_SCIENCE, params: dict = {}) -> ReportSpec:
    """Return the report configuration for the requested report type."""
    if report_type == FileTypes.L2_SCIENCE:
        return ReportSpec(
            dataset_prefix="science",
            meta_table_class=L2ScienceMetaTable,
            parquet_file_prefix="rdmt_db_{uuid}", # define prefix followed by unique ID, intentionally not an f-string
            reporting_topic_name=params[SCIENCE_REPORTING_TOPIC],
            results_table_class=L2ScienceResultsTable,
            start_time_column="exp_start_datetime",
            summary_id_column="observation_id",
        )

    elif report_type == FileTypes.L1_GUIDE_WINDOW:
        return ReportSpec(
            dataset_prefix="guide_window",
            meta_table_class=L1GuideWindowMetaTable,
            parquet_file_prefix="rdmt_gw_db_{uuid}", # define prefix followed by unique ID, intentionally not an f-string
            reporting_topic_name=params[GUIDE_WINDOW_REPORTING_TOPIC],
            results_table_class=L1GuideWindowResultsTable,
            start_time_column="acq_start_datetime",
            summary_id_column="acquisition_id",
        )

    raise ValueError(
        f"Invalid report_type '{report_type}'. Must be '{FileTypes.L2_SCIENCE}' or '{FileTypes.L1_GUIDE_WINDOW}'."
    )

def report_function(report_type: str = FileTypes.L2_SCIENCE):
    """
    Orchestrates the extraction, transformation, and reporting of RDMT file
    metadata, metrics and evaluations.

    This function connects to a MySQL database via DuckDB, identifies new
    records ready for reporting based on monitor status, updates their report
    timestamps, exports the combined metadata and results to a partitioned S3
    Parquet dataset, and finally sends a summary notification via AWS SNS.

    The process follows an ELT (Extract, Load, Transform) pattern leveraging
    DuckDB's in-memory processing and S3 connectivity.

    Parameters
    ----------
    report_type : str
        The type of report to generate either FileTypes.L2_SCIENCE or FileTypes.L1_GUIDE_WINDOW. Defaults to FileTypes.L2_SCIENCE.

    Returns
    -------
    dict
        A dictionary containing the execution status and a summary message.
        Example: ``{'statusCode': 200, 'body': [{'_message': '...'}]}``.
    """
    params = fetch_parameters_from_path(AWS_PARAMETER_PATH, expected_parameters=AWS_S3_BUCKETS + AWS_SNS_TOPICS + AWS_DBS)

    conn = None
    try:
        report_spec = _get_report_spec(report_type, params)

        logger.info('Getting SQL URL')
        sql_url = connect_to_db(database_name=params[DB_NAME], secret_name=params[DB_SECRET_NAME]).url

        # initialize duckdb local database
        logger.info('Connecting to duckdb')
        conn = duckdb.connect(database=':memory:')
        duckdb_extension_path = os.environ.get('DUCKDB_EXTENSIONS_PATH', '/duckdb_extensions')
        conn.execute(f"SET extension_directory = '{duckdb_extension_path}';")

        # Load the extensions pre-baked into your Docker image
        # 'mysql': To connect to MySQL; 'aws' & 'httpfs': To stream directly to S3
        logger.info('Loading duckdb modules')
        conn.execute("LOAD mysql;")# TODO: LOAD aws; LOAD httpfs;")

        # 2. Create a secret using the 'credential_chain' provider
        # This tells DuckDB to automatically find credentials using AWS SDK mechanisms
        logger.info('Creating duckdb secrets')
        conn.execute("CREATE SECRET (TYPE s3, PROVIDER credential_chain);")

        # --- 3. LINKING THE DATABASE ---
        # Construct a DuckDB-specific connection string from the SQLAlchemy object
        create_secret_sql = f"""
            CREATE OR REPLACE TEMPORARY SECRET db_secret (
                TYPE mysql,
                HOST '{sql_url.host}',
                DATABASE '{sql_url.database}',
                USER '{sql_url.username}',
                PASSWORD '{sql_url.password}'
            );
            """
        conn.sql(create_secret_sql)

        # 'ATTACH' creates a virtual link. It does NOT download the data yet.
        # To DuckDB, the remote MySQL database now behaves like a local schema.
        logger.info('Attaching external database to duckdb')
        conn.sql("ATTACH '' AS rdmt_db (TYPE mysql, SECRET 'db_secret');")

        # Switch to using rdmt_db server as default to allow SQLAlchemy to make SQL strings
        conn.sql("USE rdmt_db;")

        # Generate the report datetime and set the value for all columns to be reported (e.g. all rows where file reported date is not populated but all "_status" columns are -1 or 1 (indicating all monitors that needed to run have done so successfully))
        report_time = datetime.now().replace(microsecond=0)
        status_ready_condition = _status_columns_ready_condition(conn, report_spec.meta_table_name)
        update_report_time_str = f"""
            UPDATE {report_spec.meta_table_name}
            SET monitor_end_datetime = $1
            WHERE {status_ready_condition}
            AND monitor_end_datetime IS NULL
        """
        update_report_result = conn.execute(update_report_time_str, (report_time,))
        num_reported_rows = update_report_result.fetchall()[0][0]
        logger.info(f"Number of reported rows: {num_reported_rows}")

        if num_reported_rows > 0:
            # Package metadata and results data into Parquet dataset
            s3_dataset_prefix = f"{report_spec.dataset_prefix}/"
            s3_parquet_bucket_path = f"s3://{params[PARQUET_FILE_BUCKET]}/{s3_dataset_prefix}"

            # Check if we should create the parquet structure or append to it
            s3_client = boto3.client('s3')
            response = s3_client.list_objects_v2(
                Bucket=params[PARQUET_FILE_BUCKET],
                Prefix=s3_dataset_prefix,
                MaxKeys=1,
            )
            # If there are already files in this dataset prefix
            if 'Contents' in response:
                parquet_command = ", APPEND"
                logger.info('Parquet dataset exists. Appending new rows.')
            else:
                parquet_command = ""
                logger.info('Parquet dataset does not exist. Creating new dataset.')

            # TODO: We might want to sort the exported Parquet file by visit id or obs id or program to speed up queries
            parquet_export_str = f"""
                COPY (
                    SELECT
                        M.*,
                        R.* EXCLUDE(filename, reprocess_number),
                        year(M.{report_spec.start_time_column}) AS obs_year_part,
                        month(M.{report_spec.start_time_column}) AS obs_month_part
                    FROM {report_spec.meta_table_name} AS M
                    JOIN {report_spec.results_table_name} AS R
                    ON M.filename = R.filename AND M.reprocess_number = R.reprocess_number
                    WHERE M.monitor_end_datetime = $1
                )
                TO '{s3_parquet_bucket_path}'
                (FORMAT PARQUET, PARTITION_BY (obs_year_part, obs_month_part) {parquet_command}, FILENAME_PATTERN '{report_spec.parquet_file_prefix}');
            """
            conn.execute(parquet_export_str, (report_time,))
            logger.info("Saved parquet file to dataset.")

            logger.info("Generating report message string.")
            report_message = generate_report_message(
                conn,
                s3_parquet_bucket_path,
                report_time,
                report_spec,
            )

            # Send out an SNS report of all
            logger.info("Sending SNS Email notification.")
            sns = boto3.client('sns')
            report_topic = sns.create_topic(Name=report_spec.reporting_topic_name)
            try:
                response = sns.publish(
                    TopicArn=report_topic['TopicArn'],
                    Message=report_message,
                    Subject=f"RDMT Spire Report for {report_time.strftime('%Y-%m-%d')}",
                )
                logger.info('Successfully posted report to topic.')
            except Exception as e:
                logger.error(f'{e}')
                raise e

        else:
            logger.info('No new rows to report.')

            return {'statusCode': StatusCodes.SUCCESS,
                    'body': [{'message':'report_function: No new rows to report.'}]}

    except Exception as e:
        logger.error(f"{e}")
        raise e
    finally:
        if conn is not None:
            try:
                databases = conn.sql("SHOW DATABASES").fetchall()
                db_names = [row[0] for row in databases]

                if "rdmt_db" in db_names:
                    fallback_db = next((name for name in db_names if name != "rdmt_db"), None)
                    if fallback_db is not None:
                        conn.execute(f"USE {fallback_db};")
                    else:
                        # If rdmt_db is the only visible database, create one to switch into.
                        conn.execute("ATTACH ':memory:' AS fallback_db;")
                        conn.execute("USE fallback_db;")
                    conn.execute("DETACH rdmt_db;")
            except Exception as cleanup_error:
                logger.warning(f"DuckDB cleanup warning: {cleanup_error}")
            finally:
                conn.close()

    logger.info("Successfully finished reporting lambda.")
    return {'statusCode': StatusCodes.SUCCESS,
                'body': [{'_message':'Successfully completed report_function.'}]}

def _status_columns_ready_condition(conn, table_name: str) -> str:
    """Build an AND predicate requiring every *_status column to be -1 or 1.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        An active DuckDB connection object capable of executing SQL queries.
    table_name : str
        The name of the table to inspect for *_status columns.
    """
    status_columns_query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = $1
          AND column_name LIKE '%_status'
        ORDER BY column_name;
    """
    status_columns = [row[0] for row in conn.execute(status_columns_query, (table_name,)).fetchall()]

    if not status_columns:
        raise ValueError(f"No *_status columns found in table '{table_name}'.")

    # Quote identifiers in case a status column name collides with SQL keywords.
    terminal_statuses = f"{int(MonitorStatus.NOT_APPLICABLE)}, {int(MonitorStatus.COMPLETE)}"
    return " AND ".join(
        f'"{column_name}" IN ({terminal_statuses})' for column_name in status_columns
    )

def generate_report_message(
    duckdb_connection,
    s3_parquet_bucket,
    report_generation_time,
    report_spec: ReportSpec,
):
    """
    Orchestrate the creation of a RDMT SPIRE report message string.

    This function acts as the primary entry point for report generation. It
    constructs the S3 file path, calls sub-functions to identify failed
    evaluations and monitored file summaries, and wraps the results in a
    standardized header and layout.

    Parameters
    ----------
    duckdb_connection : duckdb.DuckDBPyConnection
        An active DuckDB connection object capable of executing SQL queries.
    s3_parquet_bucket : str
        The base S3 bucket path (e.g., 's3://my-data-bucket/'). The function
        appends wildcards to search for nested Parquet files.
    report_generation_time : str or datetime
        The specific timestamp used to filter the `monitor_end_datetime`
        column in the dataset.
    report_spec : ReportSpec
        A dataclass instance containing report-type-specific configuration,
        including table classes, column names, and reporting topic.

    Returns
    -------
    message : str
        The complete, formatted SPIRE report containing a timestamped header,
        a list of failed evaluations, and a hierarchical summary of
        monitored files.

    """
    s3_parquet_file_path = os.path.join(s3_parquet_bucket, "*/*/*.parquet")
    logger.info(f"Generating report message for parquet files at: {s3_parquet_file_path}")
    # Query the same subset from the results table and extract evaluations that failed

    eval_str = get_failed_evaluations(
        duckdb_connection,
        s3_parquet_file_path,
        report_generation_time,
        report_spec,
    )
    info_str = get_monitored_files(
        duckdb_connection,
        s3_parquet_file_path,
        report_generation_time,
        report_spec,
    )

    hline = "-"*40
    message = f"""
    ROMAN DATA MONITORING TOOL SPIRE REPORT

    Generated: {report_generation_time.strftime('%Y-%m-%d %H:%M:%S')}
    {hline}

    FAILED EVALUATIONS:
    {hline}
    {eval_str}

    MONITORED FILES SUMMARY:
    {hline}
    {info_str}



    """

    return message

def get_failed_evaluations(
    duckdb_connection,
    s3_parquet_file_path,
    report_generation_time,
    report_spec: ReportSpec,
):
    """
    Identify and format metric evaluations that failed (False) in a report.

    This function dynamically constructs a DuckDB SQL query to unpivot metric
    and evaluation columns from a Parquet dataset. It filters for rows where
    the evaluation result is False and returns a formatted string of the
    failures.

    Parameters
    ----------
    duckdb_connection : duckdb.DuckDBPyConnection
        An active DuckDB connection object capable of executing SQL queries.
    s3_parquet_file_path : str
        The S3 URI or path to the Parquet files (e.g., 's3://bucket/data/').
        Must support Hive-style partitioning.
    report_generation_time : str or datetime
        The specific timestamp used to filter the `monitor_end_datetime`
        column in the dataset.
    report_spec : ReportSpec
        A dataclass instance containing report-type-specific configuration,
        including table classes, column names, and reporting topic.

    Returns
    -------
    eval_str : str
        A formatted string listing failed evaluations. Each entry includes:
        - Filename
        - Reprocess number
        - Name of the failed metric
        - Value of the metric (formatted to 4 decimal places)
    """
    metric_eval_pairs = report_spec.results_table_class().get_metric_eval_pairs()
    select_clauses = "\nUNION ALL\n".join(
        [
            f"""SELECT
                filename,
                reprocess_number,
                '{m}' AS metric_name,
                CAST({m} AS VARCHAR(30)) AS metric_value,
                {e} AS evaluation
            FROM filtered_data"""
            for m, e in metric_eval_pairs
        ]
    )

    eval_check_str = f"""
        WITH filtered_data AS (
            SELECT *
            FROM read_parquet('{s3_parquet_file_path}', hive_partitioning=true, union_by_name=true)
            WHERE
                obs_year_part = YEAR({report_spec.start_time_column})
                AND obs_month_part = MONTH({report_spec.start_time_column})
                AND monitor_end_datetime = $1
        )
        SELECT
            filename,
            reprocess_number,
            metric_name,
            metric_value,
            evaluation
        FROM ({select_clauses})
        WHERE evaluation = False;
    """
    eval_result = duckdb_connection.execute(eval_check_str, (report_generation_time,)).fetchall()

    logger.info(f"Number of failed evaluations: {len(eval_result)}")

    eval_str = ""
    for row in eval_result:
        filename, rep_num, metric_name, metric_value, _ = row
        try:
            metric_value_display = f"{float(metric_value):.4f}"
        except (TypeError, ValueError):
            metric_value_display = str(metric_value)
        eval_str += (
            f"{filename} (rep # {rep_num})  >>  {metric_name} = {metric_value_display}\n"
            + tab_str
        )

    return eval_str

def get_monitored_files(
    duckdb_connection,
    s3_parquet_file_path,
    report_generation_time,
    report_spec: ReportSpec,
):
    """
    Query Parquet files from S3 and format a file summary report string.

    This function executes a DuckDB query against S3-hosted Parquet files to
    aggregate observation data (grouped by day, program, and observation ID).
    It identifies missing detectors if the count is not exactly 18 and formats
     the results into a hierarchical string structure.

    Parameters
    ----------
    duckdb_connection : duckdb.DuckDBPyConnection
        An active DuckDB connection object capable of executing SQL queries.
    s3_parquet_file_path : str
        The S3 URI or path to the Parquet files (e.g., 's3://bucket/data/').
        Must support Hive-style partitioning.
    report_generation_time : str or datetime
        The specific timestamp used to filter the `monitor_end_datetime`
        column in the dataset.
    report_spec : ReportSpec
        A dataclass instance containing report-type-specific configuration,
        including table classes, column names, and reporting topic.

    Returns
    -------
    info_str : str
        A formatted multi-line string reporting observations organized by:
        - Observation Day
            - Program Number
                - Observation ID and Reprocess Number
        Includes an "ALERT" tag for any observation with fewer than 18 detectors.
    """
    report_info_str = f"""
        SELECT
            CAST({report_spec.start_time_column} AS DATE) AS obs_day,
            reprocess_number,
            program_number,
            {report_spec.summary_id_column},
            list(detector) as detector_list,
            COUNT(detector) as detector_count,
        FROM read_parquet('{s3_parquet_file_path}', hive_partitioning=true, union_by_name=true)
        WHERE
            obs_year_part = YEAR({report_spec.start_time_column})
            AND obs_month_part = MONTH({report_spec.start_time_column})
            AND monitor_end_datetime = $1
        GROUP BY obs_day, program_number, reprocess_number, {report_spec.summary_id_column}
        ORDER BY obs_day ASC, program_number ASC, reprocess_number ASC, {report_spec.summary_id_column} ASC;
    """
    info_result = duckdb_connection.execute(report_info_str, (report_generation_time,)).fetchall()
    logger.info(f"Number of monitored file summaries: {len(info_result)}")
    logger.info(f"Monitored file summaries: {info_result}")

    info_str = ""
    print_day = None
    print_program_num = None
    for row in info_result:
        day, rep_num, program_num, obs_id, detectors, det_count = row
        if day != print_day:
            print_day = day
            print_program_num = program_num
            if len(info_str) == 0:
                info_str += f"{day}:\n"
            else:
                info_str += tab_str + f"{day}:\n"
            info_str += 2*tab_str + f"Program {program_num}:\n"
        elif program_num != print_program_num:
            print_program_num = program_num
            info_str += 2*tab_str + f"Program {program_num}:\n"
        info_str += 3*tab_str + f"{obs_id} (rep # {rep_num})"
        if det_count != 18:
            detectors.sort()
            det_list = [det.upper().removeprefix('WFI').lstrip('0') for det in detectors]
            info_str += f"  >>  ALERT: Only received detectors {" ,".join(det_list)}\n"
        else:
            info_str += "\n"

    return info_str
