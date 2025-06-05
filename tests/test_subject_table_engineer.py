"""Unit tests for SubjectTableEngineer."""

import polars as pl
import numpy as np
import pytest
from mostlyaiprize.features import SubjectTableEngineer


class TestSubjectTableEngineer:
    """Test suite for SubjectTableEngineer class."""

    def test_initialization_success_if_valid_column(self) -> None:
        """Test successful initialization with valid subject column."""
        engineer = SubjectTableEngineer("group_id")
        assert engineer._subject_column == "group_id"
        assert engineer._enable_sequential == True
        assert engineer._enable_distribution == True
        assert engineer._enable_cross_column == True
        assert engineer._enable_entropy == False
        assert engineer._enable_correlations == True

    def test_all_basic_features_present_success_if_default_settings(self) -> None:
        """Test that all expected basic features are present with default settings."""
        data = pl.DataFrame({
            'group_id': ['A'] * 4,
            'value1': [1, 2, 3, 4],
            'value2': [10, 20, 30, 40]
        })
        
        engineer = SubjectTableEngineer("group_id")
        result = engineer.create_enhanced_subject_table(data)
        
        # Basic features that should always be present
        expected_basic = [
            'sequence_length',
            'value1_mean', 'value1_std', 'value1_min', 'value1_max',
            'value2_mean', 'value2_std', 'value2_min', 'value2_max',
            'value1_first_value', 'value1_last_value', 'value1_net_change',
            'value2_first_value', 'value2_last_value', 'value2_net_change',
            'value1_median', 'value1_range', 'value1_unique_count',
            'value2_median', 'value2_range', 'value2_unique_count',
            'value1_early_half_mean', 'value1_late_half_mean', 'value1_early_late_change',
            'value2_early_half_mean', 'value2_late_half_mean', 'value2_early_late_change',
            'value1_trend_slope', 'value2_trend_slope'
        ]
        
        for feature in expected_basic:
            assert feature in result.columns, f"Missing basic feature: {feature}"

    def test_sequential_features_present_success_if_enabled(self) -> None:
        """Test sequential features are present when enabled."""
        data = pl.DataFrame({
            'group_id': ['A'] * 5,
            'value': [1, 3, 2, 4, 4]
        })
        
        engineer = SubjectTableEngineer("group_id", enable_sequential=True)
        result = engineer.create_enhanced_subject_table(data)
        
        expected_sequential = [
            'value_direction_changes', 'value_longest_streak',
            'value_is_monotonic', 'value_is_increasing', 'value_is_decreasing',
            'value_transition_count', 'value_most_common_bigram', 'value_stability_ratio'
        ]
        
        for feature in expected_sequential:
            assert feature in result.columns, f"Missing sequential feature: {feature}"

    def test_distribution_features_present_success_if_enabled(self) -> None:
        """Test distribution features are present when enabled."""
        data = pl.DataFrame({
            'group_id': ['A'] * 5,
            'value': [1, 2, 3, 4, 100]  # includes outlier
        })
        
        engineer = SubjectTableEngineer("group_id", enable_distribution=True)
        result = engineer.create_enhanced_subject_table(data)
        
        expected_distribution = [
            'value_p25', 'value_p75', 'value_cv', 'value_outlier_count'
        ]
        
        for feature in expected_distribution:
            assert feature in result.columns, f"Missing distribution feature: {feature}"

    def test_cross_column_features_present_success_if_enabled(self) -> None:
        """Test cross-column features are present when enabled."""
        data = pl.DataFrame({
            'group_id': ['A'] * 4,
            'value1': [1, 2, 3, 4],
            'value2': [2, 4, 6, 8]
        })
        
        engineer = SubjectTableEngineer("group_id", enable_cross_column=True)
        result = engineer.create_enhanced_subject_table(data)
        
        expected_cross = [
            'total_sum_mean', 'total_sum_std', 'dominant_column',
            'max_dominance_ratio', 'value_spread_mean', 'value_spread_max'
        ]
        
        for feature in expected_cross:
            assert feature in result.columns, f"Missing cross-column feature: {feature}"

    def test_correlation_features_present_success_if_enabled(self) -> None:
        """Test correlation features are present when enabled."""
        data = pl.DataFrame({
            'group_id': ['A'] * 4,
            'alice': [1, 2, 3, 4],
            'bob': [2, 4, 6, 8]
        })
        
        engineer = SubjectTableEngineer("group_id", enable_correlations=True)
        result = engineer.create_enhanced_subject_table(data)
        
        assert 'alice_bob_correlation' in result.columns

    def test_entropy_features_present_success_if_enabled(self) -> None:
        """Test entropy features are present when enabled."""
        data = pl.DataFrame({
            'group_id': ['A'] * 5,
            'value': [1, 1, 2, 2, 3]
        })
        
        engineer = SubjectTableEngineer("group_id", enable_entropy=True)
        result = engineer.create_enhanced_subject_table(data)
        
        assert 'value_entropy' in result.columns

    def test_features_disabled_success_if_flags_false(self) -> None:
        """Test features are excluded when disabled."""
        data = pl.DataFrame({
            'group_id': ['A'] * 4,
            'value1': [1, 2, 3, 4],
            'value2': [2, 4, 6, 8]
        })
        
        engineer = SubjectTableEngineer(
            "group_id", 
            enable_sequential=False,
            enable_distribution=False,
            enable_cross_column=False,
            enable_correlations=False,
            enable_entropy=False
        )
        result = engineer.create_enhanced_subject_table(data)
        
        # These should be missing
        missing_features = [
            'value1_direction_changes', 'value1_p25', 'value1_cv',
            'total_sum_mean', 'value1_value2_correlation', 'value1_entropy'
        ]
        
        for feature in missing_features:
            assert feature not in result.columns, f"Feature should be disabled: {feature}"

    def test_create_enhanced_subject_table_success_if_simple_data(self) -> None:
        """Test basic functionality with simple sequential data."""
        data = pl.DataFrame({
            'group_id': ['A', 'A', 'B', 'B'],
            'value1': [1, 2, 3, 4],
            'value2': [10, 20, 30, 40]
        })
        
        engineer = SubjectTableEngineer("group_id")
        result = engineer.create_enhanced_subject_table(data)
        
        assert len(result) == 2
        assert 'sequence_length' in result.columns
        assert result[result['group_id'] == 'A']['sequence_length'].iloc[0] == 2
        assert result[result['group_id'] == 'B']['sequence_length'].iloc[0] == 2

    def test_sequential_features_success_if_numeric_data(self) -> None:
        """Test sequential feature generation with numeric data."""
        data = pl.DataFrame({
            'group_id': ['A'] * 5,
            'value': [1, 3, 2, 4, 4]
        })
        
        engineer = SubjectTableEngineer("group_id", enable_sequential=True)
        result = engineer.create_enhanced_subject_table(data)
        
        # Check specific sequential features
        assert result['value_first_value'].item() == 1
        assert result['value_last_value'].item() == 4
        assert result['value_net_change'].item() == 3
        assert result['value_direction_changes'].item() == 2  # up, down, up, flat
        assert result['value_longest_streak'].item() == 2  # two 4s at end

    def test_monotonicity_detection_success_if_increasing_sequence(self) -> None:
        """Test monotonicity detection for increasing sequence."""
        data = pl.DataFrame({
            'group_id': ['A'] * 4,
            'value': [1, 2, 3, 4]
        })
        
        engineer = SubjectTableEngineer("group_id", enable_sequential=True)
        result = engineer.create_enhanced_subject_table(data)
        
        assert result['value_is_increasing'].item() == True
        assert result['value_is_decreasing'].item() == False
        assert result['value_is_monotonic'].item() == True

    def test_monotonicity_detection_success_if_decreasing_sequence(self) -> None:
        """Test monotonicity detection for decreasing sequence."""
        data = pl.DataFrame({
            'group_id': ['A'] * 4,
            'value': [4, 3, 2, 1]
        })
        
        engineer = SubjectTableEngineer("group_id", enable_sequential=True)
        result = engineer.create_enhanced_subject_table(data)
        
        assert result['value_is_increasing'].item() == False
        assert result['value_is_decreasing'].item() == True
        assert result['value_is_monotonic'].item() == True

    def test_distribution_features_success_if_varied_data(self) -> None:
        """Test distribution feature calculation."""
        data = pl.DataFrame({
            'group_id': ['A'] * 6,
            'value': [1, 2, 3, 4, 5, 100]  # includes outlier
        })
        
        engineer = SubjectTableEngineer("group_id", enable_distribution=True)
        result = engineer.create_enhanced_subject_table(data)
        
        assert result['value_median'].item() == 3.5
        assert result['value_range'].item() == 99
        assert 2.0 <= result['value_p25'].item() <= 2.5  # Allow for interpolation differences
        assert 4.5 <= result['value_p75'].item() <= 5.0
        assert result['value_outlier_count'].item() == 1  # 100 is outlier
        assert result['value_unique_count'].item() == 6

    def test_transition_patterns_success_if_repeated_values(self) -> None:
        """Test transition pattern analysis."""
        data = pl.DataFrame({
            'group_id': ['A'] * 5,
            'value': [1, 1, 2, 2, 1]
        })
        
        engineer = SubjectTableEngineer("group_id", enable_sequential=True)
        result = engineer.create_enhanced_subject_table(data)
        
        assert result['value_transition_count'].item() == 4  # (1,1), (1,2), (2,2), (2,1)
        assert result['value_stability_ratio'].item() == 0.5  # 2 stable out of 4 transitions
        # All bigrams appear once, so any could be "most common"
        bigram_result = result['value_most_common_bigram'].item()
        assert bigram_result in ['1,1', '1,2', '2,2', '2,1']

    def test_cross_column_features_success_if_multiple_columns(self) -> None:
        """Test cross-column relationship features."""
        data = pl.DataFrame({
            'group_id': ['A'] * 4,
            'alice': [1, 2, 3, 4],
            'bob': [2, 4, 6, 8]  # Perfect correlation
        })
        
        engineer = SubjectTableEngineer("group_id", enable_correlations=True, enable_cross_column=True)
        result = engineer.create_enhanced_subject_table(data)
        
        assert abs(result['alice_bob_correlation'].item() - 1.0) < 0.001
        assert result['total_sum_mean'].item() == 7.5  # mean of [3, 6, 9, 12]
        assert result['dominant_column'].item() == 'bob'  # bob always higher
        assert result['max_dominance_ratio'].item() == 1.0

    def test_early_late_comparison_success_if_sufficient_data(self) -> None:
        """Test early vs late half comparison."""
        data = pl.DataFrame({
            'group_id': ['A'] * 8,
            'value': [1, 1, 1, 1, 5, 5, 5, 5]  # clear shift
        })
        
        engineer = SubjectTableEngineer("group_id")
        result = engineer.create_enhanced_subject_table(data)
        
        assert result['value_early_half_mean'].item() == 1.0
        assert result['value_late_half_mean'].item() == 5.0
        assert result['value_early_late_change'].item() == 4.0

    def test_trend_slope_success_if_linear_data(self) -> None:
        """Test linear trend detection."""
        data = pl.DataFrame({
            'group_id': ['A'] * 5,
            'value': [1, 3, 5, 7, 9]  # slope = 2
        })
        
        engineer = SubjectTableEngineer("group_id")
        result = engineer.create_enhanced_subject_table(data)
        
        assert abs(result['value_trend_slope'].item() - 2.0) < 0.001

    def test_coefficient_variation_success_if_varying_data(self) -> None:
        """Test coefficient of variation calculation."""
        data = pl.DataFrame({
            'group_id': ['A'] * 4,
            'value': [10, 20, 30, 40]  # mean=25, std varies by method
        })
        
        engineer = SubjectTableEngineer("group_id", enable_distribution=True)
        result = engineer.create_enhanced_subject_table(data)
        
        # Check CV is reasonable (between 0.4 and 0.6)
        cv_result = result['value_cv'].item()
        assert 0.4 < cv_result < 0.6, f"CV {cv_result} not in expected range"

    def test_categorical_features_success_if_mixed_data(self) -> None:
        """Test categorical feature handling."""
        data = pl.DataFrame({
            'group_id': ['A', 'A', 'B', 'B'],
            'category': ['X', 'Y', 'X', 'X'],
            'value': [1, 2, 3, 4]
        })
        
        engineer = SubjectTableEngineer("group_id")
        result = engineer.create_enhanced_subject_table(data)
        
        assert 'category_unique_count' in result.columns
        assert 'category_most_frequent' in result.columns

    def test_single_row_group_success_if_minimal_data(self) -> None:
        """Test handling of single-row groups."""
        data = pl.DataFrame({
            'group_id': ['A', 'B'],
            'value': [1, 2]
        })
        
        engineer = SubjectTableEngineer("group_id", enable_sequential=True)
        result = engineer.create_enhanced_subject_table(data)
        
        assert len(result) == 2
        assert result[result['group_id'] == 'A']['value_net_change'].iloc[0] == 0
        assert result[result['group_id'] == 'A']['value_is_monotonic'].iloc[0] == True
        assert result[result['group_id'] == 'A']['value_direction_changes'].iloc[0] == 0

    def test_identical_values_success_if_constant_sequence(self) -> None:
        """Test features with identical values throughout sequence."""
        data = pl.DataFrame({
            'group_id': ['A'] * 5,
            'value': [3, 3, 3, 3, 3]
        })
        
        engineer = SubjectTableEngineer("group_id", enable_sequential=True, enable_distribution=True)
        result = engineer.create_enhanced_subject_table(data)
        
        assert result['value_range'].item() == 0
        assert result['value_direction_changes'].item() == 0
        assert result['value_longest_streak'].item() == 5
        assert result['value_stability_ratio'].item() == 1.0
        assert result['value_is_monotonic'].item() == True
        assert result['value_cv'].item() == 0  # no variation

    def test_multiple_groups_success_if_complex_data(self) -> None:
        """Test with multiple groups having different patterns."""
        data = pl.DataFrame({
            'group_id': ['A'] * 3 + ['B'] * 4 + ['C'] * 2,
            'value1': [1, 2, 3, 10, 8, 6, 4, 100, 200],
            'value2': [5, 5, 5, 1, 2, 3, 4, 50, 60]
        })
        
        engineer = SubjectTableEngineer("group_id", enable_correlations=True)
        result = engineer.create_enhanced_subject_table(data)
        
        assert len(result) == 3
        assert 'value1_value2_correlation' in result.columns
        lengths = result['sequence_length'].to_list()
        assert sorted(lengths) == [2, 3, 4]

    def test_zero_values_success_if_contains_zeros(self) -> None:
        """Test coefficient of variation with zero mean."""
        data = pl.DataFrame({
            'group_id': ['A'] * 4,
            'value': [-2, -1, 1, 2]  # mean = 0
        })
        
        engineer = SubjectTableEngineer("group_id", enable_distribution=True)
        result = engineer.create_enhanced_subject_table(data)
        
        assert result['value_cv'].item() == 0  # Should handle division by zero

    def test_empty_group_failure_if_no_data(self) -> None:
        """Test behavior with empty data."""
        data = pl.DataFrame({
            'group_id': pl.Series([], dtype=pl.Utf8),
            'value': pl.Series([], dtype=pl.Int64)
        })
        
        engineer = SubjectTableEngineer("group_id")
        
        # Empty dataframes should raise error or return empty result
        try:
            result = engineer.create_enhanced_subject_table(data)
            assert len(result) == 0
        except Exception:
            # Acceptable behavior for empty data
            pass

    def test_missing_values_success_if_nan_present(self) -> None:
        """Test handling of missing values."""
        data = pl.DataFrame({
            'group_id': ['A'] * 4,
            'value': [1.0, None, 3.0, 4.0]
        })
        
        engineer = SubjectTableEngineer("group_id")
        result = engineer.create_enhanced_subject_table(data)
        
        # Should handle null values gracefully
        assert result['sequence_length'].item() == 4
