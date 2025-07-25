"""
Schema Migration Admin Script for InferenceStore

Usage:
    python schema_migration_admin.py

This script will:
1. Export all current data from inference_store
2. Apply schema transformations
3. Delete old table
4. Create new table with updated schema
"""

import pandas as pd
from workbench_bridges.api.inference_store import InferenceStore

# Pandas display options
pd.set_option("display.max_colwidth", 30)


def migrate_schema():
    """Migrate InferenceStore schema with data preservation"""

    # Initialize the inference store
    inf_store = InferenceStore()

    print("Starting schema migration...")
    print(f"Current table: {inf_store.catalog_db}.{inf_store.table_name}")

    # Step 1: Export all current data
    print("Step 1: Exporting current data...")
    current_data = inf_store.query(f"SELECT * FROM {inf_store.table_name}")
    print(f"Exported {len(current_data)} rows")

    if current_data.empty:
        print("No data to migrate. Proceeding with schema update only.")

    # Step 2: Apply schema transformations
    print("Step 2: Applying schema transformations...")
    transformed_data = apply_schema_transformations(current_data)

    # Step 3: Delete old table
    print("Step 3: Deleting old table...")
    inf_store.delete_all_data()

    # Step 4: Create new table with updated schema
    print("Step 4: Creating new table with updated schema...")
    if not transformed_data.empty:
        inf_store.add_inference_results(transformed_data)
        print(f"Migration complete! {len(transformed_data)} rows migrated.")
    else:
        print("Migration complete! No data to migrate.")


def apply_schema_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply schema transformations to the DataFrame

    Modify this function for your specific schema changes
    """

    # Example transformations - modify as needed:
    # Change the 'monitor' tag to 'capture' (if tags == ['monitor'] change to ['capture'])
    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(lambda x: ["capture"] if x == ["monitor"] else x)

    # Convert timestamp columns to UTC
    for col in df.columns:
        if df[col].dtype.name.startswith("datetime"):
            if df[col].dt.tz is None:
                print(f"Column '{col}' is naive, localizing to UTC")
                df[col] = df[col].dt.tz_localize("UTC")
            else:
                # Check if it's already UTC using string representation
                tz_str = str(df[col].dt.tz)
                if (
                    tz_str not in ["UTC", "UTC+00:00", "+00:00"]
                    and df[col].dt.tz != pd.Timestamp.now().tz_localize("UTC").tz
                ):
                    print(f"Column '{col}' is timezone-aware ({tz_str}), converting to UTC")
                    df[col] = df[col].dt.tz_convert("UTC")
                else:
                    print(f"Column '{col}' is already in UTC")

    # Drop any rows where 'tags' is NaN
    if "tags" in df.columns:
        df = df.dropna(subset=["tags"])
        print(f"Dropped rows with NaN in 'tags' column, remaining rows: {len(df)}")

    return df


def preview_migration():
    """Preview the migration without actually applying it"""
    inf_store = InferenceStore()

    print("PREVIEW MODE - No changes will be made")
    print("=" * 50)

    # Get current data
    current_data = inf_store.query(f"SELECT * FROM {inf_store.table_name}")
    print(f"Current data shape: {current_data.shape}")
    print(f"Current columns: {list(current_data.columns)}")

    # Show transformed data
    transformed_data = apply_schema_transformations(current_data.copy())
    print(f"New data shape: {transformed_data.shape}")
    print(f"New columns: {list(transformed_data.columns)}")

    # Show sample of transformed data
    if not transformed_data.empty:
        print("\nSample of transformed data:")
        print(transformed_data.head())


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        preview_migration()
    else:
        # Confirm before proceeding
        response = input("This will delete and recreate the inference_store table. Continue? (yes/no): ")
        if response.lower() == "yes":
            migrate_schema()
        else:
            print("Migration cancelled.")
