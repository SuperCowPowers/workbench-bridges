# Sageworks Bridges
End User Application Bridges to SageWorks/AWS ML Pipelines.

## Examples
Application invocation of an Endpoint on AWS.

```
import pandas as pd

# SageWorks-Bridges Imports
from sageworks_bridges.endpoints.fast_inference import fast_inference


if __name__ == "__main__":

    # Data will be passed in from the End-User Application
    eval_df = pd.read_csv("test_evaluation_data.csv")[:1000]

    # Run inference on AWS Endpoint
    endpoint_name = "test-timing-realtime"
    results = fast_inference(endpoint_name, eval_df)

    # A Dataframe with Predictions is returned
    print(results)
```
