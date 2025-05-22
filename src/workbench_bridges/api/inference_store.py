"""InferenceStore: Manage Inference Results using AWS S3/Parquet/Snappy with Athena Queries"""

import logging
import time
import pandas as pd
import awswrangler as wr

# Workbench Bridges Imports
from workbench_bridges.aws.sagemaker_session import get_sagemaker_session
from workbench_bridges.utils.athena_utils import ensure_catalog_db, dataframe_to_table, delete_table


class InferenceStore:
    """InferenceStore: Manage Inference Results using AWS S3/Parquet/Snappy with Athena Queries

    Common Usage:
        ```python
        inf_store = InferenceStore()

        # List all Models in the Inference Store
        inf_store.models()

        # List the total number of rows in the Inference Store
        inf_store.total_rows()
        ```
    """

    def __init__(self, catalog_db: str = "inference_store", table_name: str = "inference_store"):
        """Initialize InferenceStore with S3 path and auto-register Glue/Athena table"""
        self.log = logging.getLogger("workbench-bridges")
        self.catalog_db = catalog_db
        self.table_name = table_name
        self.boto3_session = get_sagemaker_session().boto_session

        # Ensure Glue catalog DB exists
        ensure_catalog_db(catalog_db)

        # Ensure the Table exists
        if not wr.catalog.does_table_exist(self.catalog_db, self.table_name, self.boto3_session):
            self.log.error(f"Table {self.table_name} does not exist in database {self.catalog_db}.")
            self.log.error("Call `add_inference_results` to create the table.")

    def add_inference_results(self, df: pd.DataFrame):
        """Add inference results to the Inference Store

        Args:
            df (pd.DataFrame): The DataFrame containing inference results.
        """
        self.log.info(f"Adding inference results to {self.catalog_db}.{self.table_name}")
        dataframe_to_table(df, self.catalog_db, self.table_name)
        self.log.info("Inference results added successfully.")

    def total_rows(self) -> int:
        """Return the total number of rows in the Inference Store

        Returns:
            int: The total number of rows in the Inference Store.
        """
        self.log.info(f"Retrieving total rows from {self.catalog_db}.{self.table_name}")
        df = self.query(f"SELECT COUNT(*) FROM {self.table_name}")
        return df.iloc[0, 0]

    def query(self, athena_query: str) -> pd.DataFrame:
        """Run a query against the Inference Store"""
        self.log.info(f"Running query: {athena_query}")
        start_time = time.time()

        try:
            df = wr.athena.read_sql_query(
                sql=athena_query,
                database=self.catalog_db,
                ctas_approach=False,
                boto3_session=self.boto3_session,
            )
            execution_time = time.time() - start_time
            self.log.info(f"Query completed in {execution_time:.2f} seconds")
            return df
        except Exception as e:
            self.log.error(f"Failed to run query: {e}")
            return pd.DataFrame()

    def delete_all_data(self):
        """Delete all data in the Inference Store"""
        self.log.info(f"Deleting all data from {self.catalog_db}.{self.table_name}")
        delete_table(self.table_name, self.catalog_db)

    def __repr__(self):
        """Return a string representation of the InferenceStore object."""
        return f"InferenceStore(catalog_db={self.catalog_db}, table_name={self.table_name})"


if __name__ == "__main__":
    """Exercise the InferenceStore Class"""

    # Create a InferenceStore manager
    inf_store = InferenceStore()

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "compound_id": [1, 2, 3],
            "model_name": ["model1", "model2", "model3"],
            "inference_result": [0.1, 0.2, 0.3],
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        }
    )

    # Add inference results to the Inference Store
    inf_store.add_inference_results(df)

    # List the total rows
    print(f"Total rows in Inference Store: {inf_store.total_rows()}")

    # List all models
    print("Listing Models...")
    print(inf_store.query("SELECT distinct model_name FROM inference_store"))

    # Run a custom query
    print("Running custom query...")
    custom_query = "SELECT * FROM inference_store WHERE compound_id = 1"
    print(inf_store.query(custom_query))

    # Delete all data
    # inf_store.delete_all_data()
