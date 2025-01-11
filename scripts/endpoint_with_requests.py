import boto3
import requests
from requests_aws4auth import AWS4Auth
import pandas as pd
from io import StringIO

profile = "workbench_role"

# Define the region/endpoint
region = "us-west-2"
endpoint_name = "abalone-regression-end"


def invoke_endpoint_csv(endpoint_name: str, region: str, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Invoke SageMaker endpoint with CSV input and return predictions as CSV."""
    # Fetch AWS credentials
    session = boto3.Session(profile_name=profile)
    credentials = session.get_credentials().get_frozen_credentials()
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        "sagemaker",
        session_token=credentials.token,
    )

    # Convert DataFrame to CSV payload
    csv_buffer = StringIO()
    eval_df.to_csv(csv_buffer, index=False)
    payload = csv_buffer.getvalue()

    # Endpoint URL and headers
    url = f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations"
    headers = {"Content-Type": "text/csv", "Accept": "text/csv"}

    # Make the request
    response = requests.post(url, headers=headers, auth=auth, data=payload)
    response.raise_for_status()

    # Parse the response CSV into a DataFrame
    results = response.text.splitlines()
    parsed_results = [row.split(",") for row in results]
    return pd.DataFrame.from_records(parsed_results[1:], columns=parsed_results[0])


def invoke_endpoint_json(endpoint_name: str, region: str, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Invoke SageMaker endpoint with JSON input and return predictions as JSON."""
    # Fetch AWS credentials
    session = boto3.Session(profile_name=profile)
    credentials = session.get_credentials().get_frozen_credentials()
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        "sagemaker",
        session_token=credentials.token,
    )

    # Convert DataFrame to JSON payload
    payload = eval_df.to_json(orient="records")

    # Endpoint URL and headers
    url = f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Make the request
    response = requests.post(url, headers=headers, auth=auth, data=payload)
    response.raise_for_status()

    # Parse the response JSON into a DataFrame
    results = response.json()
    return pd.DataFrame(results)


if __name__ == "__main__":
    """Invoke SageMaker endpoint with both CSV and JSON formats."""

    # Note: We're using workbench here just to fetch the evaluation data
    #       You can replace this with your own data/dataframe
    from workbench.api.endpoint import Endpoint
    from workbench.utils.endpoint_utils import fs_evaluation_data

    endpoint = Endpoint(endpoint_name)
    if not endpoint.exists():
        raise ValueError(f"Endpoint {endpoint_name} does not exist.")

    # Fetch evaluation data
    eval_df = fs_evaluation_data(endpoint)

    # Test CSV input and output
    print("Testing CSV request/response...")
    csv_response_df = invoke_endpoint_csv(endpoint_name, region, eval_df)
    print("CSV Response:")
    print(csv_response_df)

    # Test JSON input and output
    print("\nTesting JSON request/response...")
    json_response_df = invoke_endpoint_json(endpoint_name, region, eval_df)
    print("JSON Response:")
    print(json_response_df)
