"""Tests for the AWS Account Stuff"""

# SageWorks-Bridge Imports
from sageworks_bridges.aws.sm_session import get_sagemaker_session


def test_sm_session():
    """Tests for the AWS Account Stuff"""

    # Get SageMaker Session
    sm_session = get_sagemaker_session()

    # List SageMaker Models
    print("\nSageMaker Models:")
    sagemaker_client = sm_session.sagemaker_client
    response = sagemaker_client.list_models()

    for model in response["Models"]:
        print(model["ModelName"])


if __name__ == "__main__":
    test_sm_session()
