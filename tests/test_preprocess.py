"""Tests for preprocessing input validation."""

import pandas as pd

from Data.preprocess import validate_dataframe


def test_validate_dataframe_clips_resource_bounds_and_preserves_other_columns():
    source = pd.DataFrame(
        {
            "cpu_pct": [-5, 55, 120],
            "ram_pct": [20, 110, 80],
            "disk_pct": [10, 20, 30],
            "label": ["Normal", "Alert", "Normal"],
        }
    )

    result = validate_dataframe(source)

    assert result["cpu_pct"].tolist() == [0, 55, 100]
    assert result["ram_pct"].tolist() == [20, 100, 80]
    assert result["disk_pct"].tolist() == [10, 20, 30]
    assert result["label"].tolist() == ["Normal", "Alert", "Normal"]
    assert source["cpu_pct"].tolist() == [-5, 55, 120]
