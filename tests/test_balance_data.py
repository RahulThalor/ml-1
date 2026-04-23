import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from balance_data import balance_data, preprocess_raw_data, save_balanced_dataset


def test_preprocess_raw_data_fills_missing_and_encodes_categories():
    df = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "Status": [0, 1, 0],
            "loan_amount": [100.0, None, 200.0],
            "loan_purpose": ["home", "car", "home"],
            "Interest_rate_spread": [0.1, 0.2, 0.3],
        }
    )

    result = preprocess_raw_data(df)

    assert "ID" not in result.columns
    assert "Interest_rate_spread" not in result.columns
    assert result["loan_amount"].isna().sum() == 0
    assert result["Status"].tolist() == [0, 1, 0]
    assert any(col.startswith("loan_purpose_") for col in result.columns)


def test_balance_data_returns_equal_class_counts():
    df = pd.DataFrame(
        {
            "Status": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "feature_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        }
    )

    balanced = balance_data(df, random_state=42)
    counts = balanced["Status"].value_counts().to_dict()

    assert counts == {0: 8, 1: 8}
    assert balanced.shape[0] == 16


def test_save_balanced_dataset_writes_csv(tmp_path: Path):
    df = pd.DataFrame({"Status": [0, 1], "feature_a": [1.0, 2.0]})
    output_file = tmp_path / "balanced.csv"

    save_balanced_dataset(df, str(output_file))

    loaded = pd.read_csv(output_file)
    pd.testing.assert_frame_equal(loaded, df)
