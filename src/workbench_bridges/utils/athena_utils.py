"""Athena Utils: Utility functions for AWS Athena."""

import sys
import logging
import pandas as pd
import awswrangler as wr
from botocore.exceptions import ClientError

# Workbench-Bridges Imports
from workbench_bridges.aws.sagemaker_session import get_sagemaker_session
from workbench_bridges.api.parameter_store import ParameterStore

log = logging.getLogger("workbench-bridges")


def ensure_catalog_db(catalog_db: str):
    """Ensure that the AWS Data Catalog Database exists"""

    # Grab a Workbench Session (this allows us to assume the Workbench ExecutionRole)
    log.info("Assuming Workbench Execution Role...")
    sagemaker_session = get_sagemaker_session()
    boto3_session = sagemaker_session.boto_session

    log.important(f"Ensuring that the AWS Data Catalog Database {catalog_db} exists...")
    try:
        wr.catalog.create_database(catalog_db, exist_ok=True, boto3_session=boto3_session)
    except ClientError as e:
        if e.response["Error"]["Code"] == "AccessDeniedException":
            log.error(f"AccessDeniedException {e}")
            log.error(f"Access denied while trying to create/access the catalog database '{catalog_db}'.")
            log.error("Create the database manually in the AWS Glue Console, or run this command:")
            log.error(f'aws glue create-database --database-input \'{{"Name": "{catalog_db}"}}\'')
            sys.exit(1)
        else:
            log.error(f"Unexpected error: {e}")
            sys.exit(1)


def dataframe_to_table(df: pd.DataFrame, database: str, table_name: str, mode: str = "append"):
    """Store a DataFrame as a Glue Catalog Table

    Args:
        df (pd.DataFrame): The DataFrame to store
        database (str): The name of the Glue Catalog database
        table_name (str): The name of the table to store
        mode (str): The mode to use when storing the DataFrame (default: "append")
    """
    log.info("Assuming Workbench Execution Role...")
    sagemaker_session = get_sagemaker_session()
    boto3_session = sagemaker_session.boto_session

    # Get the Workbench Bucket
    param_key = "/workbench/config/workbench_bucket"
    workbench_bucket = ParameterStore().get(param_key)
    if workbench_bucket is None:
        raise ValueError(f"Set '{param_key}' in Parameter Store.")

    # Create the S3 path
    s3_path = f"s3://{workbench_bucket}/athena/{database}/{table_name}/"

    # Convert timestamp columns to UTC
    for col in df.columns:
        if df[col].dtype.name.startswith("datetime"):
            if df[col].dt.tz is None:
                df[col] = df[col].dt.tz_localize("UTC")
            else:
                df[col] = df[col].dt.tz_convert("UTC")

    # Store the DataFrame as a Glue Catalog Table
    wr.s3.to_parquet(
        df=df,
        path=s3_path,
        dataset=True,
        mode=mode,
        schema_evolution=False,
        database=database,
        table=table_name,
        boto3_session=boto3_session,
    )

    # Verify that the table is created
    if wr.catalog.does_table_exist(database=database, table=table_name, boto3_session=boto3_session):
        log.info(f"Table {table_name} successfully created in database {database}.")
    else:
        log.critical(f"Failed to create table {table_name} in database {database}.")


def delete_table(table_name: str, database: str, include_s3_files: bool = True):
    """Delete a table from the Glue Catalog

    Args:
        table_name (str): The name of the table to delete
        database (str): The name of the database containing the table
        include_s3_files (bool): Whether to delete the S3 files associated with the table
    """
    log.info("Assuming Workbench Execution Role...")
    sagemaker_session = get_sagemaker_session()
    boto3_session = sagemaker_session.boto_session

    # Get the Workbench Bucket
    param_key = "/workbench/config/workbench_bucket"
    workbench_bucket = ParameterStore().get(param_key)
    if workbench_bucket is None:
        raise ValueError(f"Set '{param_key}' in Parameter Store.")

    # Create the S3 path
    s3_path = f"s3://{workbench_bucket}/athena/{database}/{table_name}/"

    # Delete the table
    wr.catalog.delete_table_if_exists(database=database, table=table_name, boto3_session=boto3_session)

    # Verify that the table is deleted
    glue_client = boto3_session.client("glue")
    try:
        glue_client.get_table(DatabaseName=database, Name=table_name)
        log.error(f"Failed to delete table {table_name} in database {database}.")
    except glue_client.exceptions.EntityNotFoundException:
        log.info(f"Table {table_name} successfully deleted from database {database}.")

    # Delete the S3 files if requested
    if include_s3_files:
        log.info(f"Deleting S3 files at {s3_path}...")
        wr.s3.delete_objects(s3_path, boto3_session=boto3_session)
        log.info(f"S3 files at {s3_path} deleted.")


if __name__ == "__main__":

    # Example usage
    my_catalog_db = "inference_store"
    ensure_catalog_db(my_catalog_db)
    print(f"Catalog database '{my_catalog_db}' exists.")

    # Example DataFrame
    df = pd.DataFrame(
        {
            "column1": [1, 2, 3],
            "column2": ["a", "b", "c"],
            "column3": [4, 5, 6],
            "column4": [7.0, 8.0, 9.0],
            "column5": [True, False, True],
        }
    )

    # Store the DataFrame as a Glue Catalog Table
    dataframe_to_table(df, my_catalog_db, "test_table")
    print(f"DataFrame stored as Glue table 'test_table' in database '{my_catalog_db}'.")

    print("Listing Tables...")
    my_boto3_session = get_sagemaker_session().boto_session
    print(list(wr.catalog.get_tables(database=my_catalog_db, boto3_session=my_boto3_session)))

    # Delete the table
    delete_table("test_table", my_catalog_db)
    print(f"Table 'test_table' deleted from database '{my_catalog_db}'.")
