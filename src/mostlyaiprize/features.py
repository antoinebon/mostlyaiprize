"""Enhanced subject table creation with feature engineering using Polars."""

import polars as pl
import pandas as pd
import numpy as np
import warnings
from typing import Any
from scipy.stats import entropy

# Suppress numpy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


class SubjectTableEngineer:
    """Create meaningful subject tables from sequential data using Polars.

    This class transforms sequential data into subject-level features by aggregating
    and engineering features from sequences grouped by a subject identifier.

    Features Added:
    ===============

    Basic Features (always included):
    - sequence_length: Number of rows per subject
    - {col}_mean, {col}_std, {col}_min, {col}_max: Standard aggregates
    - {col}_first_value, {col}_last_value, {col}_net_change: Sequence endpoints
    - {col}_median, {col}_range, {col}_unique_count: Basic distribution
    - {col}_early_half_mean, {col}_late_half_mean, {col}_early_late_change: Temporal patterns
    - {col}_trend_slope: Linear trend over sequence

    Sequential Features (enable_sequential=True):
    - {col}_direction_changes: Number of trend reversals
    - {col}_longest_streak: Max consecutive identical values
    - {col}_is_monotonic, {col}_is_increasing, {col}_is_decreasing: Monotonicity flags
    - {col}_transition_count: Unique state transitions
    - {col}_most_common_bigram: Most frequent consecutive pair
    - {col}_stability_ratio: Proportion of values equal to previous

    Distribution Features (enable_distribution=True):
    - {col}_p25, {col}_p75: Percentiles
    - {col}_cv: Coefficient of variation
    - {col}_outlier_count: Values beyond 2 standard deviations

    Cross-Column Features (enable_cross_column=True):
    - total_sum_mean, total_sum_std: Cross-column sum statistics
    - dominant_column: Column with highest values most often
    - max_dominance_ratio: Proportion of dominance
    - value_spread_mean, value_spread_max: Range between columns per row

    Correlation Features (enable_correlations=True):
    - {col1}_{col2}_correlation: Pairwise correlations between columns

    Entropy Features (enable_entropy=False, expensive):
    - {col}_entropy: Information entropy of value distribution

    Categorical Features (for non-numeric columns):
    - {col}_unique_count: Number of distinct values
    - {col}_most_frequent: Mode value

    Example:
        >>> data = pl.DataFrame({
        ...     'group_id': ['A', 'A', 'B', 'B'],
        ...     'value1': [1, 2, 3, 4],
        ...     'value2': [10, 20, 30, 40],
        ...     'category': ['X', 'Y', 'X', 'Z']
        ... })
        >>> engineer = SubjectTableEngineer('group_id')
        >>> result = engineer.create_enhanced_subject_table(data)
        >>> # Result contains ~50+ features per subject
    """

    def __init__(
        self,
        subject_column: str,
        enable_sequential: bool = True,
        enable_distribution: bool = True,
        enable_cross_column: bool = True,
        enable_entropy: bool = True,
        enable_correlations: bool = True,
    ) -> None:
        """Initialize with subject identifier column and feature flags.

        Args:
            subject_column: Column name that identifies subjects
            enable_sequential: Enable sequential pattern features
            enable_distribution: Enable distribution features (percentiles, CV, outliers)
            enable_cross_column: Enable cross-column relationship features
            enable_entropy: Enable entropy calculations (expensive)
            enable_correlations: Enable correlation calculations (expensive)
        """
        self._subject_column: str = subject_column
        self._enable_sequential: bool = enable_sequential
        self._enable_distribution: bool = enable_distribution
        self._enable_cross_column: bool = enable_cross_column
        self._enable_entropy: bool = enable_entropy
        self._enable_correlations: bool = enable_correlations

    def _compute_sequential_features(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        """Compute sequential features for a column using Polars expressions."""
        exprs = []

        # Basic sequential features (always included)
        exprs.extend(
            [
                pl.col(col).first().alias(f"{col}_first_value"),
                pl.col(col).last().alias(f"{col}_last_value"),
                (pl.col(col).drop_nulls().last() - pl.col(col).drop_nulls().first())
                .fill_null(0)
                .alias(f"{col}_net_change"),
                pl.col(col).median().alias(f"{col}_median"),
                (pl.col(col).max() - pl.col(col).min()).alias(f"{col}_range"),
                pl.col(col).n_unique().alias(f"{col}_unique_count"),
            ]
        )

        # Early vs late comparison
        exprs.extend(
            [
                pl.col(col)
                .drop_nulls()
                .slice(0, pl.col(col).drop_nulls().len() // 2)
                .mean()
                .alias(f"{col}_early_half_mean"),
                pl.col(col)
                .drop_nulls()
                .slice(pl.col(col).drop_nulls().len() // 2)
                .mean()
                .alias(f"{col}_late_half_mean"),
            ]
        )

        if self._enable_distribution:
            exprs.extend(
                [
                    pl.col(col).quantile(0.25, interpolation="linear").alias(f"{col}_p25"),
                    pl.col(col).quantile(0.75, interpolation="linear").alias(f"{col}_p75"),
                    pl.when(pl.col(col).mean() != 0)
                    .then(pl.col(col).std() / pl.col(col).mean())
                    .otherwise(0.0)
                    .alias(f"{col}_cv"),
                ]
            )

        if self._enable_sequential:
            # Monotonicity checks on non-null values
            exprs.extend(
                [
                    pl.col(col)
                    .drop_nulls()
                    .diff()
                    .drop_nulls()
                    .min()
                    .ge(0)
                    .fill_null(True)
                    .alias(f"{col}_is_increasing"),
                    pl.col(col)
                    .drop_nulls()
                    .diff()
                    .drop_nulls()
                    .max()
                    .le(0)
                    .fill_null(True)
                    .alias(f"{col}_is_decreasing"),
                ]
            )

            # Direction changes - count sign changes on non-null values
            exprs.append(
                pl.col(col)
                .drop_nulls()
                .diff()
                .drop_nulls()
                .sign()
                .diff()
                .abs()
                .sum()
                .truediv(2)
                .cast(pl.Int64)
                .fill_null(0)
                .alias(f"{col}_direction_changes")
            )

        # Compute trend slope using linear regression formula: slope = (n*Σxy - ΣxΣy) / (n*Σx² - (Σx)²)
        exprs.append(
            pl.when(pl.len() > 1)
            .then(
                (
                    pl.len() * (pl.int_range(pl.len()) * pl.col(col)).sum()
                    - pl.int_range(pl.len()).sum() * pl.col(col).sum()
                )
                / (pl.len() * (pl.int_range(pl.len()) ** 2).sum() - pl.int_range(pl.len()).sum() ** 2)
            )
            .otherwise(0.0)
            .alias(f"{col}_trend_slope")
        )

        return df.group_by(self._subject_column).agg(exprs)

    def _compute_complex_sequential_features(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        """Compute complex sequential features using pure Polars."""
        if not self._enable_sequential:
            return pl.DataFrame({self._subject_column: df[self._subject_column].unique()})

        # Use map_elements for correctness - can optimize later
        exprs = []

        # Longest streak
        def longest_streak(values):
            if len(values) <= 1:
                return len(values) if len(values) > 0 else 1

            max_streak = 1
            current_streak = 1

            for i in range(1, len(values)):
                if values[i] == values[i - 1]:
                    current_streak += 1
                else:
                    max_streak = max(max_streak, current_streak)
                    current_streak = 1

            return max(max_streak, current_streak)

        exprs.append(pl.col(col).map_elements(longest_streak, return_dtype=pl.Int64).alias(f"{col}_longest_streak"))

        # Transition count
        def transition_count(values):
            if len(values) <= 1:
                return 0
            transitions = set()
            for i in range(len(values) - 1):
                transitions.add((values[i], values[i + 1]))
            return len(transitions)

        exprs.append(pl.col(col).map_elements(transition_count, return_dtype=pl.Int64).alias(f"{col}_transition_count"))

        # Most common bigram using struct and mode
        exprs.append(
            pl.when(pl.len() > 1)
            .then(
                pl.concat_str(
                    [pl.col(col).slice(0, pl.len() - 1).cast(pl.Utf8), pl.lit(","), pl.col(col).slice(1).cast(pl.Utf8)]
                )
                .mode()
                .first()
            )
            .otherwise(pl.lit("none"))
            .fill_null("none")
            .alias(f"{col}_most_common_bigram")
        )

        # Stability ratio
        def stability_ratio(values):
            if len(values) <= 1:
                return 1.0
            stable_count = sum(1 for i in range(1, len(values)) if values[i] == values[i - 1])
            return stable_count / (len(values) - 1)

        exprs.append(pl.col(col).map_elements(stability_ratio, return_dtype=pl.Float64).alias(f"{col}_stability_ratio"))

        return df.group_by(self._subject_column).agg(exprs)

    def _compute_entropy_features(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        """Compute entropy features."""
        if not self._enable_entropy:
            return pl.DataFrame({self._subject_column: df[self._subject_column].unique()})

        def compute_entropy(values: list) -> float:
            if len(values) == 0:
                return 0.0
            from collections import Counter

            value_counts = Counter(values)
            probs = np.array(list(value_counts.values())) / len(values)
            return entropy(probs, base=2)

        return df.group_by(self._subject_column).agg(
            [pl.col(col).map_elements(compute_entropy, return_dtype=pl.Float64).alias(f"{col}_entropy")]
        )

    def _compute_distribution_features(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        """Compute advanced distribution features."""
        if not self._enable_distribution:
            return pl.DataFrame({self._subject_column: df[self._subject_column].unique()})

        def count_outliers(values: list) -> int:
            if len(values) <= 1:
                return 0
            values_arr = np.array(values)
            mean_val = np.mean(values_arr)
            std_val = np.std(values_arr)
            if std_val == 0:
                return 0
            z_scores = np.abs((values_arr - mean_val) / std_val)
            return int(np.sum(z_scores > 2))

        return df.group_by(self._subject_column).agg(
            [pl.col(col).map_elements(count_outliers, return_dtype=pl.Int64).alias(f"{col}_outlier_count")]
        )

    def _compute_correlations(self, df: pl.DataFrame, numerical_cols: list[str]) -> pl.DataFrame:
        """Compute pairwise correlations between columns."""
        if not self._enable_correlations or len(numerical_cols) < 2:
            return df.group_by(self._subject_column).agg([])

        exprs = []
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i + 1 :]:
                exprs.append(pl.corr(pl.col(col1), pl.col(col2)).fill_null(0).alias(f"{col1}_{col2}_correlation"))

        return df.group_by(self._subject_column).agg(exprs)

    def _compute_cross_column_features(self, df: pl.DataFrame, numerical_cols: list[str]) -> pl.DataFrame:
        """Compute cross-column relationship features."""
        if not self._enable_cross_column or len(numerical_cols) < 2:
            return df.group_by(self._subject_column).agg([])

        # Row sums and statistics using fold
        sum_expr = pl.fold(
            acc=pl.lit(0), function=lambda acc, x: acc + x, exprs=[pl.col(col) for col in numerical_cols]
        )

        exprs = [
            sum_expr.mean().alias("total_sum_mean"),
            sum_expr.std().fill_null(0).alias("total_sum_std"),
        ]

        # Value spreads (max - min per row) using fold
        max_expr = pl.fold(
            acc=pl.col(numerical_cols[0]),
            function=lambda acc, x: pl.when(x > acc).then(x).otherwise(acc),
            exprs=[pl.col(col) for col in numerical_cols[1:]],
        )
        min_expr = pl.fold(
            acc=pl.col(numerical_cols[0]),
            function=lambda acc, x: pl.when(x < acc).then(x).otherwise(acc),
            exprs=[pl.col(col) for col in numerical_cols[1:]],
        )

        exprs.extend(
            [
                (max_expr - min_expr).mean().alias("value_spread_mean"),
                (max_expr - min_expr).max().alias("value_spread_max"),
            ]
        )

        # Dominant column (most frequent argmax)
        def compute_dominance(group_df: pl.DataFrame) -> tuple[str, float]:
            """Compute which column dominates and by what ratio."""
            if len(numerical_cols) < 2:
                return numerical_cols[0] if numerical_cols else "none", 1.0

            col_data = group_df.select(numerical_cols)
            argmax_indices = []
            for row in col_data.iter_rows():
                # Filter out None values
                valid_vals = [(i, val) for i, val in enumerate(row) if val is not None]
                if not valid_vals:
                    continue  # Skip rows with all None values
                max_idx, max_val = max(valid_vals, key=lambda x: x[1])
                argmax_indices.append(numerical_cols[max_idx])

            if not argmax_indices:
                return numerical_cols[0], 1.0

            from collections import Counter

            counts = Counter(argmax_indices)
            most_common = counts.most_common(1)[0]
            return most_common[0], most_common[1] / len(argmax_indices)

        # Compute dominance for each group
        dominance_data = []
        for subject_id, group_df in df.group_by(self._subject_column):
            dom_col, dom_ratio = compute_dominance(group_df)
            dominance_data.append(
                {
                    self._subject_column: subject_id[0] if isinstance(subject_id, tuple) else subject_id,
                    "dominant_column": dom_col,
                    "max_dominance_ratio": dom_ratio,
                }
            )

        base_result = df.group_by(self._subject_column).agg(exprs)
        dominance_df = pl.DataFrame(dominance_data)

        return base_result.join(dominance_df, on=self._subject_column)

    def create_enhanced_subject_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced subject table with aggregated features.

        Args:
            data: Sequential data with subject identifiers (Polars or Pandas DataFrame)

        Returns:
            Enhanced subject table with one row per subject (Pandas DataFrame)
        """
        # Convert pandas to polars if needed
        data = pl.from_pandas(data)

        # Basic sequence length
        subject_table = data.group_by(self._subject_column).agg([pl.len().alias("sequence_length")])

        # Get column types
        numerical_cols = [
            col for col, dtype in data.schema.items() if dtype.is_numeric() and col != self._subject_column
        ]
        categorical_cols = [
            col for col, dtype in data.schema.items() if not dtype.is_numeric() and col != self._subject_column
        ]

        # Standard numerical aggregates
        if numerical_cols:
            standard_aggs = []
            for col in numerical_cols:
                standard_aggs.extend(
                    [
                        pl.col(col).mean().alias(f"{col}_mean"),
                        pl.col(col).std().alias(f"{col}_std"),
                        pl.col(col).min().alias(f"{col}_min"),
                        pl.col(col).max().alias(f"{col}_max"),
                    ]
                )

            numerical_stats = data.group_by(self._subject_column).agg(standard_aggs)
            subject_table = subject_table.join(numerical_stats, on=self._subject_column)

        # Categorical aggregates
        if categorical_cols:
            cat_aggs = []
            for col in categorical_cols:
                cat_aggs.extend(
                    [
                        pl.col(col).n_unique().alias(f"{col}_unique_count"),
                        pl.col(col).mode().first().alias(f"{col}_most_frequent"),
                    ]
                )

            categorical_stats = data.group_by(self._subject_column).agg(cat_aggs)
            subject_table = subject_table.join(categorical_stats, on=self._subject_column)

        # Add enhanced features for each numerical column
        for col in numerical_cols:
            # Basic sequential features
            basic_features = self._compute_sequential_features(data, col)
            subject_table = subject_table.join(basic_features, on=self._subject_column)

            # Complex sequential features (only join if has features)
            complex_features = self._compute_complex_sequential_features(data, col)
            if complex_features.width > 1:  # Has columns other than subject_column
                subject_table = subject_table.join(complex_features, on=self._subject_column)

            # Distribution features (only join if has features)
            dist_features = self._compute_distribution_features(data, col)
            if dist_features.width > 1:  # Has columns other than subject_column
                subject_table = subject_table.join(dist_features, on=self._subject_column)

            # Entropy features (only join if has features)
            entropy_features = self._compute_entropy_features(data, col)
            if entropy_features.width > 1:  # Has columns other than subject_column
                subject_table = subject_table.join(entropy_features, on=self._subject_column)

        # Cross-column features
        if len(numerical_cols) >= 2:
            # Correlations
            if self._enable_correlations:
                corr_features = self._compute_correlations(data, numerical_cols)
                if corr_features.width > 1:
                    subject_table = subject_table.join(corr_features, on=self._subject_column)

            # Cross-column relationships
            cross_features = self._compute_cross_column_features(data, numerical_cols)
            if cross_features.width > 1:
                subject_table = subject_table.join(cross_features, on=self._subject_column)

        # Add computed columns for early/late change
        for col in numerical_cols:
            if f"{col}_early_half_mean" in subject_table.columns and f"{col}_late_half_mean" in subject_table.columns:
                subject_table = subject_table.with_columns(
                    [
                        (pl.col(f"{col}_late_half_mean") - pl.col(f"{col}_early_half_mean")).alias(
                            f"{col}_early_late_change"
                        )
                    ]
                )

        # Add monotonic flags
        for col in numerical_cols:
            if self._enable_sequential and f"{col}_is_increasing" in subject_table.columns:
                subject_table = subject_table.with_columns(
                    [(pl.col(f"{col}_is_increasing") | pl.col(f"{col}_is_decreasing")).alias(f"{col}_is_monotonic")]
                )

        # Convert back to pandas before returning
        return subject_table.to_pandas()


__all__ = ["SubjectTableEngineer"]
