import os
import sys
import pandas as pd

# SageWorks-Bridges Imports
from sageworks_bridges.endpoints.fast_inference import fast_inference


if __name__ == "__main__":

    # Check if we have local data
    if not os.path.exists("test_evaluation_data.csv"):
        print("Download Data to test_evaluation_data.csv")
        sys.exit(1)

    # Data will be passed in from the End-User Application
    eval_df = pd.read_csv("test_evaluation_data.csv")[:1000]

    # Run inference on AWS Endpoint
    endpoint_name = "test-timing-realtime"
    results = fast_inference(endpoint_name, eval_df)

    # A Dataframe with Predictions is returned
    print(results)
