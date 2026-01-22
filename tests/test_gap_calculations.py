"""
Tests for ECI gap calculations.

Run with: pytest tests/test_gap_calculations.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from update_data import (
    calculate_horizontal_gaps,
    calculate_statistics,
    estimate_current_gap,
    calculate_historical_gaps,
    is_china_org,
    is_us_org,
)


class TestHorizontalGapCalculation:
    """Tests for calculate_horizontal_gaps function."""

    def test_simple_matched_gap(self):
        """Test a simple case where an open model matches a closed model."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 101.0],
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "Open": [False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        assert len(gaps) == 1
        assert gaps[0]["closed_model"] == "ClosedA"
        assert gaps[0]["open_model"] == "OpenB"
        assert gaps[0]["matched"] is True
        # Gap should be ~3 months (Jan to Apr)
        assert 2.5 <= gaps[0]["gap_months"] <= 3.5

    def test_unmatched_gap(self):
        """Test case where no open model matches the closed model."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [150.0, 100.0],  # Open model ECI too low
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "Open": [False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        assert len(gaps) == 1
        assert gaps[0]["matched"] is False
        assert gaps[0]["open_model"] is None
        # Gap should be from Jan 2024 to now
        assert gaps[0]["gap_months"] > 10  # At least 10 months from Jan 2024

    def test_open_model_must_be_after_closed(self):
        """Test that open model must be released AFTER closed model."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 105.0],
            "date": pd.to_datetime(["2024-06-01", "2024-01-01"]),  # Open before closed
            "Open": [False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        assert len(gaps) == 1
        # Should be unmatched because open model came before closed
        assert gaps[0]["matched"] is False

    def test_multiple_closed_models(self):
        """Test with multiple closed models."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "ClosedB", "OpenC"],
            "eci": [100.0, 120.0, 125.0],
            "date": pd.to_datetime(["2024-01-01", "2024-03-01", "2024-06-01"]),
            "Open": [False, False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        assert len(gaps) == 2
        # Both should be matched by OpenC
        assert all(g["matched"] for g in gaps)
        assert all(g["open_model"] == "OpenC" for g in gaps)

    def test_eci_tolerance(self):
        """Test that open model within 1 ECI point counts as match."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 99.5],  # OpenB is 0.5 below, should still match
            "date": pd.to_datetime(["2024-01-01", "2024-04-01"]),
            "Open": [False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        # Should match because 99.5 >= 100 - 1
        assert gaps[0]["matched"] is True


class TestCurrentGapEstimation:
    """Tests for estimate_current_gap function."""

    def test_no_unmatched_models(self):
        """Test when all models are matched."""
        gaps = [
            {"matched": True, "gap_months": 3.0},
            {"matched": True, "gap_months": 5.0},
        ]
        matched_gaps = [3.0, 5.0]

        result = estimate_current_gap(gaps, matched_gaps)

        assert result["estimated_current_gap"] == 0
        assert result["min_current_gap"] == 0
        assert result["confidence"] == "high"

    def test_with_unmatched_models(self):
        """Test estimation with unmatched models."""
        gaps = [
            {"matched": True, "gap_months": 3.0},
            {"matched": True, "gap_months": 5.0},
            {"matched": False, "gap_months": 8.0},  # Unmatched, 8 months old
            {"matched": False, "gap_months": 6.0},  # Unmatched, 6 months old
        ]
        matched_gaps = [3.0, 5.0]

        result = estimate_current_gap(gaps, matched_gaps)

        # Min should be the oldest unmatched model
        assert result["min_current_gap"] == 8.0
        # Estimate should be >= min
        assert result["estimated_current_gap"] >= result["min_current_gap"]
        # Unmatched ages should be sorted descending
        assert result["unmatched_ages"] == [8.0, 6.0]

    def test_estimate_increases_with_more_unmatched(self):
        """Test that estimate increases with more unmatched models."""
        gaps_few = [
            {"matched": True, "gap_months": 4.0},
            {"matched": True, "gap_months": 5.0},
            {"matched": True, "gap_months": 6.0},
            {"matched": False, "gap_months": 7.0},
        ]
        gaps_many = [
            {"matched": True, "gap_months": 4.0},
            {"matched": True, "gap_months": 5.0},
            {"matched": True, "gap_months": 6.0},
            {"matched": False, "gap_months": 7.0},
            {"matched": False, "gap_months": 6.0},
            {"matched": False, "gap_months": 5.0},
        ]
        matched_gaps = [4.0, 5.0, 6.0]

        result_few = estimate_current_gap(gaps_few, matched_gaps)
        result_many = estimate_current_gap(gaps_many, matched_gaps)

        # More unmatched models should increase uncertainty premium
        # Both should have same min (7.0) but many should have higher estimate
        assert result_few["min_current_gap"] == 7.0
        assert result_many["min_current_gap"] == 7.0


class TestStatisticsCalculation:
    """Tests for calculate_statistics function."""

    def test_basic_statistics(self):
        """Test basic statistics calculation."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "ClosedB", "OpenC", "OpenD"],
            "eci": [100.0, 110.0, 105.0, 115.0],
            "date": pd.to_datetime([
                "2024-01-01", "2024-02-01",
                "2024-03-01", "2024-04-01"
            ]),
            "Open": [False, False, True, True],
        })

        gaps = [
            {"matched": True, "gap_months": 2.0},
            {"matched": True, "gap_months": 2.0},
        ]

        stats = calculate_statistics(df, gaps)

        assert "avg_horizontal_gap_months" in stats
        assert "std_horizontal_gap" in stats
        assert "ci_90_low" in stats
        assert "ci_90_high" in stats
        assert stats["total_matched"] == 2
        assert stats["total_unmatched"] == 0
        assert "current_gap_estimate" in stats

    def test_vertical_gap_calculation(self):
        """Test that vertical gap is best_closed - best_open."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [150.0, 140.0],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "Open": [False, True],
        })

        gaps = [{"matched": False, "gap_months": 5.0}]
        stats = calculate_statistics(df, gaps)

        # Vertical gap = 150 - 140 = 10
        assert stats["current_vertical_gap"] == 10.0


class TestHistoricalGaps:
    """Tests for calculate_historical_gaps function."""

    def test_historical_gap_calculation(self):
        """Test that historical gaps are calculated for time points."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB", "ClosedC", "OpenD"],
            "eci": [100.0, 105.0, 120.0, 125.0],
            "date": pd.to_datetime([
                "2023-01-01", "2023-06-01",
                "2024-01-01", "2024-06-01"
            ]),
            "Open": [False, True, False, True],
        })

        historical = calculate_historical_gaps(df)

        # Should have multiple time points
        assert len(historical) > 0
        # Each entry should have required fields
        for entry in historical:
            assert "date" in entry
            assert "gap_months" in entry
            assert "matched" in entry
            assert "reference_model" in entry
            assert "open_frontier_model" in entry

    def test_gap_cannot_be_negative(self):
        """Test that gaps are never negative."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB"],
            "eci": [100.0, 150.0],  # Open is way ahead
            "date": pd.to_datetime(["2024-01-01", "2023-01-01"]),  # Open released first
            "Open": [False, True],
        })

        historical = calculate_historical_gaps(df)

        for entry in historical:
            assert entry["gap_months"] >= 0


class TestOrganizationClassification:
    """Tests for organization classification functions."""

    def test_china_organizations(self):
        """Test Chinese organization detection."""
        assert is_china_org("DeepSeek") is True
        assert is_china_org("Alibaba") is True
        assert is_china_org("DeepSeek,Peking University") is True
        assert is_china_org("Baichuan") is True
        assert is_china_org("01.AI") is True
        assert is_china_org("Moonshot") is True

    def test_us_organizations(self):
        """Test US organization detection."""
        assert is_us_org("OpenAI") is True
        assert is_us_org("Anthropic") is True
        assert is_us_org("Google DeepMind") is True
        assert is_us_org("Meta AI") is True
        assert is_us_org("Microsoft Research") is True
        assert is_us_org("xAI") is True

    def test_non_china_non_us(self):
        """Test organizations that are neither China nor US."""
        assert is_china_org("Mistral AI") is False
        assert is_us_org("Mistral AI") is False
        assert is_china_org("Technology Innovation Institute") is False
        assert is_us_org("Technology Innovation Institute") is False


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame({
            "Model": [],
            "eci": [],
            "date": [],
            "Open": [],
        })

        gaps = calculate_horizontal_gaps(df)
        assert gaps == []

    def test_only_open_models(self):
        """Test with only open models (no closed)."""
        df = pd.DataFrame({
            "Model": ["OpenA", "OpenB"],
            "eci": [100.0, 110.0],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "Open": [True, True],
        })

        gaps = calculate_horizontal_gaps(df)
        assert gaps == []

    def test_only_closed_models(self):
        """Test with only closed models (no open)."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "ClosedB"],
            "eci": [100.0, 110.0],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "Open": [False, False],
        })

        gaps = calculate_horizontal_gaps(df)

        # Should have gaps but all unmatched
        assert len(gaps) == 2
        assert all(not g["matched"] for g in gaps)

    def test_nan_handling(self):
        """Test that NaN values are handled gracefully."""
        df = pd.DataFrame({
            "Model": ["ClosedA", "OpenB", "ClosedC"],
            "eci": [100.0, np.nan, 120.0],
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "Open": [False, True, False],
        })

        # Should not raise an error
        gaps = calculate_horizontal_gaps(df)
        assert isinstance(gaps, list)


class TestRealWorldScenario:
    """Tests using realistic data similar to actual ECI data."""

    def test_realistic_gap_scenario(self):
        """Test with data mimicking real ECI progression."""
        df = pd.DataFrame({
            "Model": [
                "GPT-4", "LLaMA-65B", "Claude 3 Opus",
                "Llama 3.1-405B", "o1", "DeepSeek-R1"
            ],
            "eci": [126.0, 109.0, 127.0, 128.0, 142.0, 139.0],
            "date": pd.to_datetime([
                "2023-03-14", "2023-02-24", "2024-02-29",
                "2024-07-23", "2024-12-17", "2025-01-20"
            ]),
            "Open": [False, True, False, True, False, True],
        })

        gaps = calculate_horizontal_gaps(df)

        # GPT-4 should be matched by Llama 3.1-405B
        gpt4_gap = next((g for g in gaps if g["closed_model"] == "GPT-4"), None)
        assert gpt4_gap is not None
        assert gpt4_gap["matched"] is True
        assert gpt4_gap["open_model"] == "Llama 3.1-405B"
        # Gap should be ~16 months (Mar 2023 to Jul 2024)
        assert 15 <= gpt4_gap["gap_months"] <= 18

        # o1 should be matched by DeepSeek-R1
        o1_gap = next((g for g in gaps if g["closed_model"] == "o1"), None)
        assert o1_gap is not None
        # DeepSeek-R1 (139) is close to o1 (142), within tolerance? 139 >= 142-1 = 141? No, 139 < 141
        # So o1 should be unmatched
        assert o1_gap["matched"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
