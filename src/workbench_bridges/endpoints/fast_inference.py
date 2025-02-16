"""Fast Inference on SageMaker Endpoints"""

import pandas as pd
from io import StringIO

# Sagemaker Imports
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker import Predictor


def fast_inference(endpoint_name: str, eval_df: pd.DataFrame, sagemaker_session=None) -> pd.DataFrame:
    """Run inference on the Endpoint using the provided DataFrame

    Args:
        endpoint_name (str): The name of the Endpoint
        eval_df (pd.DataFrame): The DataFrame to run predictions on
        sagemaker_session (sagemaker.session.Session): The SageMaker Session (optional)

    Returns:
        pd.DataFrame: The DataFrame with predictions

    Note:
        There's no sanity checks or error handling... just FAST Inference!
    """
    predictor = Predictor(
        endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=CSVSerializer(),
        deserializer=CSVDeserializer(),
    )

    # Convert the DataFrame into a CSV buffer
    csv_buffer = StringIO()
    eval_df.to_csv(csv_buffer, index=False)

    # Send the CSV Buffer to the predictor
    results = predictor.predict(csv_buffer.getvalue())

    # Construct a DataFrame from the results
    results_df = pd.DataFrame.from_records(results[1:], columns=results[0])
    return results_df


if __name__ == "__main__":
    """Exercise the Endpoint Utilities"""
    from workbench.api.endpoint import Endpoint
    from workbench.utils.endpoint_utils import fs_evaluation_data

    # Grab the Endpoint
    my_endpoint_name = "abalone-regression-end"
    my_endpoint = Endpoint(my_endpoint_name)
    if not my_endpoint.exists():
        print(f"Endpoint {my_endpoint_name} does not exist.")
        exit(1)

    # Pull evaluation data
    print("Pulling Evaluation Data...")
    sagemaker_session = my_endpoint.sm_session
    my_eval_df = fs_evaluation_data(my_endpoint)

    # Run Fast Inference
    print("Starting Fast Inference...")
    my_results_df = fast_inference(my_endpoint_name, my_eval_df, sagemaker_session)
    print(my_results_df)
