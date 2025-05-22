"""Athena Utils: Utility functions for AWS Athena."""

import sys
import logging
import pandas as pd
import awswrangler as wr
from botocore.exceptions import ClientError

# Workbench-Bridges Imports
from workbench_bridges.aws.sagemaker_session import get_sagemaker_session

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


def dataframe_to_table(df: pd.DataFrame, s3_path: str, database: str, table_name: str):
    """Store a DataFrame as a Glue Catalog Table

    Args:
        df (pd.DataFrame): The DataFrame to store
        s3_path (str): The S3 path to store the DataFrame
        database (str): The name of the Glue Catalog database
        table_name (str): The name of the table to store
    """
    log.info("Assuming Workbench Execution Role...")
    sagemaker_session = get_sagemaker_session()
    boto3_session = sagemaker_session.boto_session

    # Store the DataFrame as a Glue Catalog Table
    wr.s3.to_parquet(
        df=df,
        path=s3_path,
        dataset=True,
        mode="overwrite",
        database=database,
        table=table_name,
        boto3_session=boto3_session,
    )

    # Verify that the table is created
    glue_client = boto3_session.client("glue")
    try:
        glue_client.get_table(DatabaseName=database, Name=table_name)
        log.info(f"Table {table_name} successfully created in database {database}.")
    except glue_client.exceptions.EntityNotFoundException:
        log.critical(f"Failed to create table {table_name} in database {database}.")


def delete_table(table_name: str, database: str, boto3_session):
    """Delete a table from the Glue Catalog

    Args:
        table_name (str): The name of the table to delete
        database (str): The name of the database containing the table
        boto3_session: The boto3 session
    """

    # Delete the table
    wr.catalog.delete_table_if_exists(database=database, table=table_name, boto3_session=boto3_session)

    # Verify that the table is deleted
    glue_client = boto3_session.client("glue")
    try:
        glue_client.get_table(DatabaseName=database, Name=table_name)
        log.error(f"Failed to delete table {table_name} in database {database}.")
    except glue_client.exceptions.EntityNotFoundException:
        log.info(f"Table {table_name} successfully deleted from database {database}.")


def upsert_glue_table_not_used(
    catalog_db: str,
    table_name: str,
    s3_path: str,
    columns_types: dict,
    partition_cols: list[str] | None = None,
):
    """Ensure that the Glue table exists"""
    log.info("Assuming Workbench Execution Role...")
    sagemaker_session = get_sagemaker_session()
    boto3_session = sagemaker_session.boto_session

    log.important(f"Ensuring Glue table {catalog_db}.{table_name} exists...")

    wr.catalog.create_parquet_table(
        database=catalog_db,
        table=table_name,
        path=s3_path,
        columns_types=columns_types,
        partition_cols=partition_cols or [],
        boto3_session=boto3_session,
        mode="overwrite",
    )


if __name__ == "__main__":

    # Example usage
    my_catalog_db = "inference_store"
    ensure_catalog_db(my_catalog_db)
    print(f"Catalog database '{my_catalog_db}' exists.")
