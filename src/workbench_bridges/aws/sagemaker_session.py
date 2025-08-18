"""Get an AWS SageMaker Session"""

import boto3
import sagemaker
import logging
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError

# Workbench Bridges Imports
from workbench_bridges.utils.execution_environment import running_as_service

# Set up logging
log = logging.getLogger("workbench-bridges")


def get_sagemaker_session() -> sagemaker.Session:
    # Create initial SageMaker session
    session = sagemaker.Session()
    # Only assume Workbench role when running locally (not as a service)
    if not running_as_service():
        role = "Workbench-ExecutionRole"
        account_id = session.boto_session.client("sts").get_caller_identity()["Account"]
        try:
            assumed_role = boto3.client("sts").assume_role(
                RoleArn=f"arn:aws:iam::{account_id}:role/{role}", RoleSessionName="WorkbenchSession"
            )
            credentials = assumed_role["Credentials"]
            session.boto_session = boto3.Session(
                aws_access_key_id=credentials["AccessKeyId"],
                aws_secret_access_key=credentials["SecretAccessKey"],
                aws_session_token=credentials["SessionToken"],
            )
        except (ClientError, NoCredentialsError, PartialCredentialsError) as e:
            # Log the failure and proceed with the default session
            log.important(f"Failed to assume Workbench role: {e}. Using default session.")
    return session


if __name__ == "__main__":
    from workbench_bridges.api.parameter_store import ParameterStore

    # Get SageMaker Session
    sagemaker_session = get_sagemaker_session()

    # List SageMaker Models
    print("\nSageMaker Models:")
    sagemaker_client = sagemaker_session.sagemaker_client
    response = sagemaker_client.list_models()

    for model in response["Models"]:
        print(model["ModelName"])

    # Get the boto3 session and list objects in the Workbench Bucket
    param_key = "/workbench/config/workbench_bucket"
    workbench_bucket = ParameterStore().get(param_key)
    if workbench_bucket is None:
        raise ValueError(f"Set '{param_key}' in Parameter Store.")
    boto3_session = sagemaker_session.boto_session
    s3_client = boto3_session.client("s3")
    try:
        response = s3_client.list_objects_v2(Bucket=workbench_bucket, MaxKeys=10)
        if "Contents" in response:
            print(f"\nFirst 10 objects in '{workbench_bucket}':")
            for obj in response["Contents"]:
                print(f"  {obj['Key']} ({obj['Size']} bytes)")
        else:
            print(f"\nBucket '{workbench_bucket}' is empty or no objects found.")
    except ClientError as e:
        print(f"Failed to access workbench bucket '{workbench_bucket}': {e}")
