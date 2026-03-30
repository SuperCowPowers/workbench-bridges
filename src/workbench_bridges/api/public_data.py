"""PublicData: Read-only access to public S3 data (comp_chem datasets)"""

from typing import Union, Optional
import json
import logging
import awswrangler as wr
import pandas as pd
import boto3
from botocore import UNSIGNED
from botocore.config import Config

from workbench_bridges.utils.aws_utils import not_found_returns_none


class PublicData:
    """PublicData: Read-only list/get interface for public S3 datasets

    Common Usage:
        ```python
        public_data = PublicData()

        # List available datasets
        public_data.list()

        # Get a specific dataset
        df = public_data.get("aqsol/aqsol_public_data")
        print(df)
        ```
    """

    # Public bucket and prefix
    BUCKET = "workbench-public-data"
    PREFIX = "comp_chem/"

    def __init__(self):
        """PublicData Init Method"""
        self.log = logging.getLogger("workbench-bridges")

        # Anonymous boto3 session and config (no credentials needed for public data)
        self.boto3_session = boto3.Session(region_name="us-west-2")
        self.unsigned_config = Config(signature_version=UNSIGNED)
        self.s3_client = self.boto3_session.client("s3", config=self.unsigned_config)

    def list(self) -> list:
        """List all available datasets

        Returns:
            list: Dataset names (relative paths without extensions) available in the public store.
        """
        datasets = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.BUCKET, Prefix=self.PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key == self.PREFIX or obj["Size"] == 0:
                    continue
                name = key[len(self.PREFIX) :]
                # Strip file extensions (.csv, .parquet, etc.)
                for ext in (".parquet", ".csv", ".json"):
                    if name.endswith(ext):
                        name = name[: -len(ext)]
                        break
                datasets.append(name)

        return sorted(datasets)

    @not_found_returns_none
    def get(self, name: str) -> Union[pd.DataFrame, None]:
        """Retrieve a dataset by name

        Args:
            name (str): The dataset name (as returned by list()).

        Returns:
            pd.DataFrame: The retrieved DataFrame or None if not found.
        """
        # Set unsigned config for awswrangler
        wr.config.botocore_config = self.unsigned_config

        # Try parquet first, then csv
        for ext, reader in [(".parquet", wr.s3.read_parquet), (".csv", wr.s3.read_csv)]:
            s3_uri = f"s3://{self.BUCKET}/{self.PREFIX}{name}{ext}"
            try:
                df = reader(s3_uri, boto3_session=self.boto3_session)
                self.log.info(f"Reading {s3_uri}...")
                return df
            except wr.exceptions.NoFilesFound:
                continue

        self.log.warning(f"Dataset '{name}' not found in public data store.")
        return None

    def details(self) -> pd.DataFrame:
        """Return detailed metadata for all datasets

        Returns:
            pd.DataFrame: DataFrame with name, size (MB), and modified date for each dataset.
        """
        rows = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.BUCKET, Prefix=self.PREFIX):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key == self.PREFIX or obj["Size"] == 0:
                    continue
                name = key[len(self.PREFIX) :]
                rows.append(
                    {
                        "name": name,
                        "size (MB)": round(obj["Size"] / (1024 * 1024), 2),
                        "modified": obj["LastModified"].strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["name", "size (MB)", "modified"])

    def describe(self, name: str) -> Optional[dict]:
        """Return a description of a dataset including source references.

        Looks up the dataset in the descriptions.json file stored alongside
        the data in S3 (s3://<bucket>/<prefix>descriptions.json).

        Args:
            name: Dataset filename (e.g. "logp_all.csv") or dataset name
                  without extension (e.g. "logp_all").

        Returns:
            dict with description, column info, references, etc., or None if not found.
        """
        # Load descriptions from S3 (cached after first call)
        if not hasattr(self, "_descriptions"):
            self._descriptions = self._load_descriptions()

        # Build candidate keys: exact, with extensions, and basename variants
        import posixpath

        basename = posixpath.basename(name)
        # Strip extension from basename if present
        stem = basename
        for ext in (".parquet", ".csv", ".json"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break

        candidates = [name, basename, stem, f"{stem}.csv", f"{stem}.parquet", f"{basename}.csv", f"{basename}.parquet"]
        for key in candidates:
            if key in self._descriptions:
                return self._descriptions[key]

        self.log.info(f"No description found for '{name}'")
        return None

    def _load_descriptions(self) -> dict:
        """Load descriptions.json from S3."""
        s3_key = f"{self.PREFIX}descriptions.json"
        try:
            resp = self.s3_client.get_object(Bucket=self.BUCKET, Key=s3_key)
            return json.loads(resp["Body"].read().decode("utf-8"))
        except Exception as e:
            self.log.info(f"Could not load descriptions from s3://{self.BUCKET}/{s3_key}: {e}")
            return {}

    def __repr__(self):
        """Return a string representation of the PublicData object."""
        details_df = self.details()
        if details_df.empty:
            return "PublicData: No datasets found."

        max_name_len = details_df["name"].str.len().max() + 2
        details_df["name"] = details_df["name"].str.ljust(max_name_len)
        details_df["size (MB)"] = details_df["size (MB)"].apply(lambda x: f"{x:.2f} MB")
        details_df["modified"] = details_df["modified"].apply(lambda x: f" ({x})")
        return details_df.to_string(index=False, header=False)


if __name__ == "__main__":
    """Exercise the PublicData Class"""

    public_data = PublicData()

    # List datasets
    print("Available Datasets:")
    print(public_data.list())

    # Details
    print("\nDataset Details:")
    print(public_data.details())

    # Repr
    print("\nPublicData Object:")
    print(public_data)

    # Get a dataset
    datasets = public_data.list()
    if datasets:
        print(f"\nGetting first dataset: {datasets[0]}")
        df = public_data.get(datasets[0])
        print(df)
