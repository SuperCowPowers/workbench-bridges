"""Get an AWS SageMaker Session"""

import boto3
import sagemaker


def get_sagemaker_session() -> sagemaker.Session:
    # Create initial SageMaker session
    session = sagemaker.Session()

    # Specify the Workbench execution role
    role = "Workbench-ExecutionRole"
    account_id = session.boto_session.client("sts").get_caller_identity()["Account"]

    try:
        # Attempt to assume the Workbench execution role
        assumed_role = boto3.client("sts").assume_role(
            RoleArn=f"arn:aws:iam::{account_id}:role/{role}", RoleSessionName="WorkbenchSession"
        )

        # Update the boto session in SageMaker session with assumed role credentials
        credentials = assumed_role["Credentials"]
        session.boto_session = boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )
    except Exception as e:
        # Log the failure and proceed with the default session
        print(f"Failed to assume Workbench role, this is probably fine...")

    return session


if __name__ == "__main__":

    # Get SageMaker Session
    sagemaker_session = get_sagemaker_session()

    # List SageMaker Models
    print("\nSageMaker Models:")
    sagemaker_client = sagemaker_session.sagemaker_client
    response = sagemaker_client.list_models()

    for model in response["Models"]:
        print(model["ModelName"])

    # Get the boto3 session and list S3 buckets
    boto3_session = sagemaker_session.boto_session
    s3_client = boto3_session.client("s3")
    response = s3_client.list_buckets()
    if "Buckets" in response:
        print("\nS3 Buckets:")
        for bucket in response["Buckets"]:
            print(bucket["Name"])
