import io
import json
import logging
import os
import shutil
import uuid
from pathlib import Path

import boto3

from rdmt_spire.constants.codes import StatusCodes
from rdmt_spire.constants.lambdas import (
    ALEMBIC_VERSIONS_BUCKET,
    AWS_DBS,
    AWS_PARAMETER_PATH,
    AWS_S3_BUCKETS,
)
from rdmt_spire.utilities.aws_utils import (
    fetch_parameters_from_path,
    remove_from_s3,
    write_to_s3,
)
from rdmt_spire.utilities.db_utils import (
    download_migration_scripts_from_s3,
    run_alembic_downgrade,
    run_alembic_upgrade,
)

# Set up logging to capture logs in a string stream for later retrieval
logger = logging.getLogger()
#for old_handler in logger.handlers:
#    logger.removeHandler(old_handler)
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
codepipeline_client = boto3.client('codepipeline')

def handler(event, context):
    """Lambda function handler run to update the Spire database in AWS."""
    try:
        params = fetch_parameters_from_path(AWS_PARAMETER_PATH, expected_parameters=AWS_DBS + AWS_S3_BUCKETS)
        alembic_file_local_path = Path('/tmp/alembic_files')

        if 'CodePipeline.job' in event:
            codepipeline_job_id = event["CodePipeline.job"]["id"]
            user_parameters = event["CodePipeline.job"]["data"]["actionConfiguration"]["configuration"]["UserParameters"]
            user_parameters_dict = json.loads(user_parameters)
            change_type = user_parameters_dict.get("change_type")
            allow_update = user_parameters_dict.get("allow_update")
            revision_str = user_parameters_dict.get("revision_str")
            manual_revision = user_parameters_dict.get('manual_revision')
            log_bucket_name = user_parameters_dict.get('log_bucket_name')
        else:
            codepipeline_job_id = None

            change_type = event.get("change_type")
            allow_update = event.get("allow_update")
            revision_str = event.get("revision_str")
            manual_revision = event.get('manual_revision')
            log_bucket_name = event.get('log_bucket_name')

        log_stream = None
        new_handler = None
        if log_bucket_name:
            log_stream = io.StringIO()
            new_handler = logging.StreamHandler(log_stream)
            formatter = logging.Formatter('%(asctime)s - %(levelname)-7.7s [%(name)s] %(message)s')
            new_handler.setFormatter(formatter)
            logger.addHandler(new_handler)
            logger.setLevel(logging.INFO)

        if manual_revision:
            logger.info('manual_revision = True thus saving revision file, but manual upgrade using revision string will be needed.')
            allow_update = False
        else:
            manual_revision = False # handle case where manual_revision is not specified (default of .get() is None)

        if allow_update:
            logger.info("allow_update = True thus any changes will be pushed to the database.")
        elif allow_update is None:
            logger.warning("allow_update not set by input message. Using default (False).")
            allow_update = False
        else:
            logger.info("allow_update = False thus no changes will be executed.")

        # Download version files and prep the alembic directory for changes
        try:
            logger.info(f"Downloading version files from: {params[ALEMBIC_VERSIONS_BUCKET]} to {alembic_file_local_path}")
            download_migration_scripts_from_s3(params[ALEMBIC_VERSIONS_BUCKET], alembic_file_local_path, s3_client=s3_client)
            logger.info("Download complete.")

            env_path = "/var/task/alembic_files/env.py"
            mako_path = "/var/task/alembic_files/script.py.mako" 

            logger.info(f'Copying required alembic files to {alembic_file_local_path}')
            shutil.copy(env_path, alembic_file_local_path)
            shutil.copy(mako_path, alembic_file_local_path)
            logger.info('Copy complete.')
        except Exception as e:
            logger.error(f"{e.__class__.__name__}: {str(e)}")
            check_codepipeline_return(codepipeline_job_id, StatusCodes.ALEMBIC_FAILED_VERSIONS_DOWNLOAD)
            return {
                'statusCode': StatusCodes.ALEMBIC_FAILED_VERSIONS_DOWNLOAD,
                'body': json.dumps({
                    'message': f'Could not access/download the alembic version files from s3 bucket({params[ALEMBIC_VERSIONS_BUCKET]}).',
                    'error': f"{e.__class__.__name__}: {str(e)}"
                })
            }

        # if attempting to push the database history forward
        if change_type == "upgrade":
            try:
                logger.info('Starting alembic checking/upgrade process.')
                db_update_bool, revision_path = run_alembic_upgrade(
                    'alembic.ini',
                    str(alembic_file_local_path),
                    revision_str,
                    allow_update=allow_update,
                    save_revision=manual_revision,
                    app_logger=logger,
                )
                logger.info('alembic executions complete.')
            except Exception as e:
                logger.error(f"{e.__class__.__name__}: {str(e)}")
                check_codepipeline_return(codepipeline_job_id, StatusCodes.ALEMBIC_UPDATE_FAILURE)
                return {
                    'statusCode': StatusCodes.ALEMBIC_UPDATE_FAILURE,
                    'body': json.dumps({
                        'message': 'Could not upgrade the database tables successfully.',
                        'error': f"{e.__class__.__name__}: {str(e)}"
                    })
                }

            if db_update_bool and (revision_path is not None) and (allow_update or manual_revision):
                # TODO: add an try-except here to re-downgrade and remove upgrade if writing the version file fails
                rev_object_name = os.path.basename(revision_path)
                write_to_s3(revision_path, params[ALEMBIC_VERSIONS_BUCKET], rev_object_name, s3_client=s3_client)

            # prepare an exit message
            if db_update_bool and allow_update:
                exit_message = 'Successfully updated the database tables using alembic.'
            elif db_update_bool and manual_revision:
                exit_message = 'Successfully configured database update using alembic, revision saved as manual_revision = True, not upgraded as allow_update = False.'
            elif db_update_bool:
                exit_message = 'Successfully configured database update using alembic but not upgraded as allow_update = False.'
            else:
                exit_message = 'Database does not need updates.'

        # if attempting to revert to a previous database change
        elif change_type == "downgrade":
            try:
                logger.info('Starting alembic downgrade process.')
                versions_to_delete = run_alembic_downgrade(
                    'alembic.ini',
                    str(alembic_file_local_path),
                    revision_str,
                    allow_update=allow_update,
                    app_logger=logger,
                )
                logger.info('alembic executions complete.')
            except Exception as e:
                logger.error(f"{e.__class__.__name__}: {str(e)}")
                check_codepipeline_return(codepipeline_job_id, StatusCodes.ALEMBIC_UPDATE_FAILURE)
                return {
                    'statusCode': StatusCodes.ALEMBIC_UPDATE_FAILURE,
                    'body': json.dumps({
                        'message': 'Could not downgrade the database tables successfully.',
                        'error': f"{e.__class__.__name__}: {str(e)}"
                    })
                }
            
            if versions_to_delete and allow_update:
                # TODO: think about this implementation more
                for revision in versions_to_delete:
                    rev_object_name = f"{os.path.basename(revision.path)}"
                    remove_from_s3(params[ALEMBIC_VERSIONS_BUCKET], rev_object_name, s3_client=s3_client)

            # prepare an exit message
            if versions_to_delete and allow_update:
                exit_message = 'Successfully updated the database tables using alembic.'
            elif versions_to_delete:
                exit_message = 'Successfully configured database update using alembic but not downgraded as allow_update = False.'
            else:
                exit_message = 'Database not updated.'

        logger.info(f'alembic_handler finished: {exit_message}')

        # save the logging file to s3 if a log bucket name was provided
        if log_bucket_name and log_stream is not None:
            log_identifier = codepipeline_job_id or uuid.uuid4().hex
            file_key = f'alembic_handler_logs/{log_identifier}_log.txt'

            # Write the log stream to S3
            s3_client.put_object(
                Bucket=log_bucket_name,
                Key=file_key,
                Body=log_stream.getvalue(),
                ContentType='text/plain'
            )

            review_log_url = s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': log_bucket_name, 'Key': file_key},
                ExpiresIn=1800 # 30 minutes in seconds
            )
            logger.info(f'Log file saved to s3://{log_bucket_name}/{file_key}')
        else:
            review_log_url = None
            logger.info('No log bucket name provided, thus no log file saved to S3.')

        if new_handler is not None:
            logger.removeHandler(new_handler)
            new_handler.close()

        check_codepipeline_return(codepipeline_job_id, StatusCodes.SUCCESS, log_file_url=review_log_url)
        logger.info("Finished alembic_handler execution.")
        response = {
            'statusCode': StatusCodes.SUCCESS,
            'message': exit_message,
        }
        if codepipeline_job_id is None:
            response['review_log_url'] = review_log_url

        return response
    
    except Exception as e:
        logger.error(f"{e.__class__.__name__}: {str(e)}")
        return {
            'statusCode': StatusCodes.ALEMBIC_UPDATE_FAILURE,
            'body': json.dumps({
                'message': 'An unexpected error occurred in the alembic_handler.',
                'error': f"{e.__class__.__name__}: {str(e)}"
            })
        }

def check_codepipeline_return(job_id, status, log_file_url=None):
    """Check the status of a CodePipeline job and update it accordingly.
    
    Parameters
    ----------
    job_id : str
        The ID of the CodePipeline job to check.
    status : int
        The status code to return to CodePipeline.
    log_file_url : str, optional
        The URL of the log file to include in the job success result.

    """
    if job_id is not None:
        if (status == StatusCodes.SUCCESS) and (log_file_url is not None):
            logger.info(f"CodePipeline job {job_id} succeeded. Log file URL: {log_file_url}")
            try:
                codepipeline_client.put_job_success_result(
                    jobId = job_id,
                    outputVariables = {
                        'review_log_url': log_file_url,
                    }
                )
            except Exception as e:
                logger.error(f"Failed to update CodePipeline job {job_id} success result: {e}")
                codepipeline_client.put_job_failure_result(
                    jobId = job_id,
                    failureDetails = {
                        'type': 'JobFailed',
                        'message': "Failed in alembic_handler. See logs for details.",
                    }
                )
        else:
            if status == StatusCodes.SUCCESS and (log_file_url is None):
                failure_message = f"CodePipeline job {job_id} succeeded but no log file URL was provided."
                logger.error(failure_message)
            else:
                failure_message = f"CodePipeline job {job_id} failed with status {status}."
                logger.error(failure_message)

            codepipeline_client.put_job_failure_result(
                    jobId = job_id,
                    failureDetails = {
                        'type': 'JobFailed',
                        'message': failure_message,
                    }
                )