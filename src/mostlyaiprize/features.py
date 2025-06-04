"""Enhanced subject table creation with feature engineering from sequential data."""

import pandas as pd
import numpy as np
from typing import Any

class SubjectTableEngineer:
    """Create meaningful subject tables from sequential data."""
    
    def __init__(self, subject_column: str) -> None:
        """Initialize with subject identifier column.
        
        Args:
            subject_column: Column name that identifies subjects
        """
        self._subject_column: str = subject_column
    
    def create_enhanced_subject_table(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced subject table with aggregated features.
        
        Args:
            data: Sequential data with subject identifiers
            
        Returns:
            Enhanced subject table with one row per subject
        """
        # Group by subject for aggregations
        grouped = data.groupby(self._subject_column)
        
        # Initialize with basic stats
        subject_table = grouped.size().reset_index(name='sequence_length')
        
        # Add numerical aggregates
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            numerical_stats = grouped[numerical_cols].agg(['mean', 'std', 'min', 'max']).reset_index()
            # Flatten column names properly
            new_cols = [self._subject_column]
            for col in numerical_cols:
                new_cols.extend([f"{col}_mean", f"{col}_std", f"{col}_min", f"{col}_max"])
            numerical_stats.columns = new_cols
            subject_table = subject_table.merge(numerical_stats, on=self._subject_column)
        
        # Add categorical stats
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        categorical_cols = [col for col in categorical_cols if col != self._subject_column]
        if len(categorical_cols) > 0:
            def safe_mode(x):
                mode_result = x.mode()
                return mode_result.iloc[0] if len(mode_result) > 0 else None
            
            categorical_stats = grouped[categorical_cols].agg(['nunique', safe_mode]).reset_index()
            # Flatten column names properly  
            new_cols = [self._subject_column]
            for col in categorical_cols:
                new_cols.extend([f"{col}_unique_count", f"{col}_most_frequent"])
            categorical_stats.columns = new_cols
            subject_table = subject_table.merge(categorical_stats, on=self._subject_column)
        
        return subject_table


__all__ = ["SubjectTableEngineer"]
