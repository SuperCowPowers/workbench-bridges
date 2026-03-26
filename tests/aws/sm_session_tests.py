"""Tests for the AWS Session"""

# Workbench-Bridge Imports
from workbench_bridges.aws.sagemaker_session import get_boto3_session


def test_boto3_session():
    """Tests for the AWS boto3 session and SageMaker client access"""

    # Get boto3 Session
    boto3_session = get_boto3_session()

    # List SageMaker Models
    print("\nSageMaker Models:")
    sagemaker_client = boto3_session.client("sagemaker")
    response = sagemaker_client.list_models()

    for model in response["Models"]:
        print(model["ModelName"])


if __name__ == "__main__":
    test_boto3_session()
