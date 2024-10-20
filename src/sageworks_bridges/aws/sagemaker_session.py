"""Get an AWS SageMaker Session"""

import boto3
import sagemaker


def get_sagemaker_session() -> sagemaker.Session:

    # Create a SageMaker session
    session = sagemaker.Session()

    # Get the SageMaker role
    role = "SageWorks-ExecutionRole"

    # Attach the role to the session
    boto3.client("sts").assume_role(
        RoleArn=f'arn:aws:iam::{session.boto_session.client("sts").get_caller_identity()["Account"]}:role/{role}',
        RoleSessionName="SageWorksSession",
    )

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
