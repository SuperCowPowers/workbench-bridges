"""Tests for the PublicData functionality"""

import pandas as pd

# Workbench-Bridges Imports
from workbench_bridges.api.public_data import PublicData

# Create a PublicData object
pub_data = PublicData()


def test_repr():
    print("PublicData Object...")
    print(pub_data)


def test_list():
    datasets = pub_data.list()
    print(f"Available datasets: {datasets}")
    assert isinstance(datasets, list)
    assert len(datasets) > 0


def test_details():
    details = pub_data.details()
    print(f"Dataset Details:\n{details}")
    assert isinstance(details, pd.DataFrame)
    assert not details.empty
    assert "name" in details.columns
    assert "size (MB)" in details.columns
    assert "modified" in details.columns


def test_get_parquet():
    # Get a known dataset (parquet format)
    df = pub_data.get("comp_chem/aqsol/aqsol_public_data")
    print(f"Got aqsol dataset: {df.shape}")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_get_csv():
    # Get a known csv dataset
    df = pub_data.get("comp_chem/logp/logp_all")
    print(f"Got logp dataset: {df.shape}")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_get_not_found():
    result = pub_data.get("nonexistent_dataset_xyz_123")
    assert result is None


def test_describe():
    datasets = pub_data.list()
    assert len(datasets) > 0
    desc = pub_data.describe(datasets[0])
    print(f"Description for '{datasets[0]}': {desc}")
    # Description may or may not exist, just verify no crash
    assert desc is None or isinstance(desc, dict)


def test_no_awswrangler_config_pollution():
    """Verify that PublicData.get() does not pollute awswrangler global config.

    This was the root cause of MissingAuthenticationTokenException in batch jobs:
    PublicData.get() previously set wr.config.botocore_config to UNSIGNED globally,
    which caused all subsequent awswrangler calls to strip auth signatures.
    """
    import awswrangler as wr

    # Capture config state before
    config_before = wr.config.botocore_config

    # Call get (this previously poisoned the global config)
    pub_data.get("comp_chem/aqsol/aqsol_public_data")

    # Verify config is unchanged
    config_after = wr.config.botocore_config
    assert (
        config_before == config_after
    ), f"PublicData.get() modified wr.config.botocore_config: {config_before} -> {config_after}"


if __name__ == "__main__":
    test_repr()
    test_list()
    test_details()
    test_get_parquet()
    test_get_csv()
    test_get_not_found()
    test_describe()
    test_no_awswrangler_config_pollution()
    print("\nAll tests passed!")
