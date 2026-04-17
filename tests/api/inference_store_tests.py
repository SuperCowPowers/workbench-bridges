"""Read-only tests for the Workbench InferenceStore functionality"""

import logging
import numbers
import pandas as pd

# Workbench-Bridges Imports
from workbench_bridges.api.inference_store import InferenceStore

# Show debug calls
logging.getLogger("workbench-bridges").setLevel(logging.DEBUG)


def test_instantiation():
    """InferenceStore should instantiate with expected defaults."""
    inf_store = InferenceStore()
    assert inf_store.catalog_db == "inference_store"
    assert inf_store.table_name == "inference_store"
    assert inf_store.schema == ["id", "model", "pred_label", "pred_value", "tags", "meta", "timestamp"]
    assert repr(inf_store) == "InferenceStore(catalog_db=inference_store, table_name=inference_store)"


def test_total_rows():
    """total_rows() should return a non-negative integer."""
    inf_store = InferenceStore()
    rows = inf_store.total_rows()
    print(f"Total rows: {rows} (type={type(rows).__name__})")
    assert isinstance(rows, numbers.Real)  # Accepts Python int/float and numpy scalars
    assert int(rows) >= 0


def test_query_returns_dataframe():
    """A simple SELECT against the store should return a DataFrame with expected columns."""
    inf_store = InferenceStore()
    df = inf_store.query(f"SELECT * FROM {inf_store.table_name} LIMIT 1")
    assert isinstance(df, pd.DataFrame)
    # If there are rows, they should contain schema columns
    if not df.empty:
        for col in ("id", "model", "timestamp"):
            assert col in df.columns, f"Expected column '{col}' missing in query result"


if __name__ == "__main__":
    test_instantiation()
    test_total_rows()
    test_query_returns_dataframe()
