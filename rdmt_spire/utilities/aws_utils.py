import io
import os
from pathlib import Path
from typing import Any, Dict, List

import boto3
import numpy as np
from astropy.table import QTable
from botocore.exceptions import (
    ClientError,
)
from dotenv import dotenv_values

from ..constants.lambdas import AWS_PARAMETER_PATH


def get_monitor_config():
    """Fetch the monitor configuration from either AWS Parameter Store or a local .env file.

    Returns
    -------
    dict
        Dictionary containing the monitor configuration parameters.
    """
    config={}
    config["RDMT_SPIRE_LDATA_DIR"] = str(Path(__file__).resolve().parent.parent / "rdmt_data")
    config["RDMT_SPIRE_RDATA_DIR"] = ""
    try:
        expected_parameters=["RDMT_SPIRE_L4_BUCKET", "RDMT_SPIRE_RDATA_BUCKET"]
        result= fetch_parameters_from_path(AWS_PARAMETER_PATH, expected_parameters)
        for key in result:
            config[key.removesuffix("_BUCKET")+'_DIR'] = f"s3://{result[key]}"
    except KeyError as e:
        print(f"Error fetching monitor config from AWS Parameter Store: {e}")
        raise e
    except Exception:
        expected_parameters=["RDMT_SPIRE_L4_DIR", "RDMT_SPIRE_RDATA_DIR", "RDMT_SPIRE_L2_DIR"]
        result = dotenv_values('.env')
        missing_params = set(expected_parameters) - set(result.keys())
        if missing_params:
            # We do not raise an exception here because
            # want to conditionally select or skip unit tests based on the available configuration
            print("Error fetching monitor config from .env file.", missing_params)
            # raise e
        for key in expected_parameters:
            if key in result:
                config[key]=result[key]
    return config


def fetch_parameters_from_path(path: str, expected_parameters: List[str], ssm_client=None) -> Dict[str, str]:
    """Fetch parameters from AWS Parameter Store at a given path.
    
    Retrieves specified parameters under a given path in AWS Systems Manager 
    Parameter Store and returns them as a dictionary.
    
    Parameters
    ----------
    path : str
        The parameter path in AWS Parameter Store (e.g., '/app/config/prod').
    expected_parameters : list of str
        List of expected parameter names to fetch from the path.
    ssm_client : boto3.client, optional
        An existing boto3 SSM client instance. Defaults to None, in which case 
        a new client is created.
    
    Returns
    -------
    dict
        Dictionary with parameter names as keys and their values from Parameter Store.
    """
    if ssm_client is None:
        ssm_client = boto3.client('ssm', region_name='us-east-1')
    
    params_dict = {}
 
    # Paginate through all parameters under the given path
    paginator = ssm_client.get_paginator('get_parameters_by_path')
    page_iterator = paginator.paginate(
        Path=path,
        Recursive=True,
        WithDecryption=True
    )
    
    for page in page_iterator:
        for parameter in page['Parameters']:
            # Extract parameter name and remove the path prefix
            param_name = parameter['Name']
            # Remove leading path separator(s) and the path prefix
            if param_name.startswith(path):
                param_short_name = param_name[len(path):].lstrip('/')
            else:
                param_short_name = param_name
            
            # Add to dictionary if it's in the expected parameters
            if param_short_name in expected_parameters:
                params_dict[param_short_name] = parameter['Value']
                
    missing_params = set(expected_parameters) - set(params_dict.keys())
    if missing_params:
        raise KeyError(f"Missing expected SSM parameters from path '{path}': {missing_params}")
    
    return params_dict


def load_s3_object(bucket_name:str, key_name:str, config=None):
    """Load an object (file) from an S3 bucket on AWS.
    
    Parameters
    ----------
    bucket_name : str
        path to the bucket that is storing the S3 object
    key_name : str
        equivalent to the filename of the object in the S3 bucket for retrieval
    
    Returns
    -------
    the contents of the file as a series of bytes

    """
    s3_client = boto3.client('s3', config=config)
    response = s3_client.get_object(
        Bucket=bucket_name,
        Key=key_name
    )

    content = io.BytesIO(response["Body"].read())

    return content

def make_sqs_message_attributes(msg_attrs:Dict[str,Any]) -> Dict[str,Any]:
    """Create a message attribute for an SQS message.
    
    Parameters
    ----------
    msg_attrs : dict
        Collection of attributes to add to a message
    
    Returns
    -------
    dictionary with the correct formatting to attach the desired attributes to the JSON SQS message.
    
    """
    sqs_msg_attrs = {}
    for key, value in msg_attrs.items():
        if isinstance(value, (np.floating, float)):
            dtype="Number.float"
        elif isinstance(value, (np.integer, int)):
            dtype="Number.int"
        elif isinstance(value, str):
            dtype="String"
        else:
            dtype="String"
            
        sqs_msg_attrs[key] = {"DataType": dtype, "StringValue": str(value)}

    return sqs_msg_attrs

def write_to_s3(local_filepath:str, s3_bucket:str, object_name:str, s3_client=None):
    """
    Upload a local file to an Amazon S3 bucket.

    This function uploads a file from the local filesystem to the specified
    S3 bucket and object key. If no S3 client is provided, a default boto3
    S3 client will be created.

    Parameters
    ----------
    local_filepath : str
        The path to the local file to upload.
    s3_bucket : str
        The name of the target S3 bucket.
    object_name : str
        The S3 object key (file name in the bucket).
    s3_client : boto3.client, optional
        An existing boto3 S3 client instance.
            Defaults to None, in which case a new client is created.

    Example:
        >>> write_to_s3("data/version.txt", "my-bucket", "version.txt")
        'version.txt' successfully uploaded to 'my-bucket'.

    """
    if s3_client is None:
        s3_client = boto3.client('s3')
    # save the version file to s3
    try:
        s3_client.upload_file(local_filepath, s3_bucket, object_name)
        print(f"'{object_name}' successfully uploaded to '{s3_bucket}'.")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        raise

def remove_from_s3(s3_bucket, object_name, s3_client=None):
    """
    Delete an object from an Amazon S3 bucket.

    This function removes a specified object from an S3 bucket using the AWS SDK for Python (boto3).
    If no S3 client is provided, a new client instance will be created.

    Parameters
    ----------
    s3_bucket : str
        The name of the S3 bucket from which the object should be deleted.
    object_name : str
        The key (file name) of the object to delete from the bucket.
    s3_client : boto3.client, optional
        An existing boto3 S3 client instance. If None, a new client will be created.

    Example
    -------
        >>> remove_from_s3("my-bucket", "path/to/file.txt")
        'path/to/file.txt' successfully deleted from 'my-bucket'.

    """
    if s3_client is None:
        s3_client = boto3.client('s3')
    # save the version file to s3
    try:
        s3_client.delete_object(Bucket=s3_bucket, Key=object_name)
        print(f"'{object_name}' successfully deleted from '{s3_bucket}'.")
    except Exception as e:
        print(f"Error removing file from S3: {e}")
        raise

def get_sqs_url(queue_name, account_id=None, sqs_client=None):
    """
    Resolve and return the AWS SQS queue URL for a given queue name.

    This helper fetches the SQS queue URL by name. If an AWS account ID is not
    provided, it will call STS to determine the account of the active caller.
    You may optionally pass an existing `boto3` SQS client to control client
    lifecycle and configuration (e.g., region, credentials, retries). If no
    client is supplied, the function creates a temporary SQS client and closes it.

    Parameters
    ----------
    queue_name : str
        The name of the SQS queue (without URL).
    account_id : str, optional
        The AWS account ID that owns the queue. If not provided, the function
        uses STS `GetCallerIdentity` to determine the current caller's account.
    sqs_client : boto3.client, optional
        An existing `boto3` SQS client. If omitted, a new client is created for
        the duration of the call and then closed.

    Returns
    -------
    str
        The full SQS queue URL corresponding to `queue_name`.
    """
    # If the caller didn't specify an account_id, use STS to determine it
    if account_id is None:
        sts_client = boto3.client('sts')
        response = sts_client.get_caller_identity()
        account_id = response['Account']
        sts_client.close()
    
    # Track whether we created the SQS client so we can close it later
    close_client = False
    if sqs_client is None:
        sqs_client = boto3.client('sqs')
        close_client = True

    # Ask SQS for the queue URL by name and owning account ID
    response = sqs_client.get_queue_url(
            QueueName=queue_name,
            QueueOwnerAWSAccountId=account_id
        )
    
    # If we created the client in this function, close it to free resources
    if close_client:
        sqs_client.close()

    return response['QueueUrl']


def load_file_object(bucket_name: str, key_name: str, mode: str = "rb", config=None):
    """Load a file form a local filesystem or from an S3 bucket on AWS.

    Parameters
    ----------
    bucket_name : str
        path to the S3 bucket (or local directory) that is storing the file
        If the path starts with "s3://", it is treated as an S3 bucket; otherwise, 
        it is treated as a local directory.
    key_name : str
        equivalent to the filename of the object in the S3 bucket for retrieval
    config : botocore.config.Config, optional
        Needed for anonymous access to public S3 buckets, by setting:
            from botocore import UNSIGNED
            from botocore.config import Config
            config=Config(signature_version=UNSIGNED)

    Returns
    -------
    the contents of the file as a series of bytes

    """
    if bucket_name.startswith("s3://"):
        content = load_s3_object(bucket_name[len("s3://"):], key_name, config=config)
    else:
        local_path = os.path.join(bucket_name, key_name)
        if os.path.exists(local_path):
            with open(local_path, mode=mode) as fp:
                content = io.BytesIO(fp.read())
        else:
            raise FileNotFoundError(f"File not found in S3 bucket or local path: {local_path}")        

    return content


def file_exists(bucket_name: str, key_name: str, config=None):
    """
    Check if a file exists in a local filesystem or in an S3 bucket on AWS

    Parameters
    ----------
    bucket_name : str
        path to the S3 bucket (or local directory) that is storing the file.
        If the path starts with "s3://", it is treated as an S3 bucket; otherwise, 
        it is treated as a local directory.
    key_name : str
        equivalent to the filename of the object in the S3 bucket for retrieval
    config : botocore.config.Config, optional
        Needed for anonymous access to public S3 buckets, by setting:
    Returns
    -------
    bool

    """
    if bucket_name.startswith("s3://"):
        s3_client = boto3.client("s3", config=config)
        try:
            s3_client.head_object(Bucket=bucket_name[len("s3://"):], Key=key_name)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            else:
                # Something else went wrong (e.g., permissions)
                raise e
    else:
        return os.path.exists(os.path.join(bucket_name, key_name))


def csv2qtable(bucket_name, filename: str, lower: bool = True, config=None) -> QTable:
    """
    Load a CSV or ECSV file from a local directory or an S3 bucket and return it as a QTable.

    Parameters
    ----------
    bucket_name : str
        path to the S3 bucket (or local directory) that is storing the file.
        If the path starts with "s3://", it is treated as an S3 bucket; otherwise, 
        it is treated as a local directory.
    filename : str
        name of the file to be loaded.
    lower : bool, optional
        If True, convert all column names to lowercase. Default is True.
    config : botocore.config.Config, optional
        Needed for anonymous access to public S3 buckets, by setting:
    """
    file_object = load_file_object(bucket_name, filename, config=config)
    if filename.endswith(".ecsv"):
        content = file_object.read().decode("utf-8")
        df = QTable.read(content.replace("\t", " "), format="ascii.ecsv")
    else:
        df = QTable.read(file_object, format="ascii.csv")

    # use lowercase for all column names
    for col in df.colnames:
        df.rename_column(col, col.strip().lower())
    return df

