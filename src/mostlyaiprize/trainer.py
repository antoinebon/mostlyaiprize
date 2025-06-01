"""Simplified trainer for MOSTLY AI Prize Challenge 2."""

import logging
import time
from typing import Any

import mlflow
import pandas as pd
from mostlyai.sdk import MostlyAI
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class Trainer:
    """Simplified trainer for Challenge 2 with config-driven approach."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize trainer with configuration.
        
        Args:
            config: Configuration object
        """
        self._config: DictConfig = config
        self._mostly: MostlyAI = MostlyAI(local=True)
        
        # Setup MLflow
        mlflow.set_experiment(config.experiment_name)
    
    def _build_mostly_config(self, data: pd.DataFrame) -> dict[str, Any]:
        """Build MOSTLY AI configuration from Hydra config and data.
        
        Args:
            data: Training data
            
        Returns:
            MOSTLY AI configuration dict
        """
        # Start with base generator config
        mostly_config = {
            'name': self._config.generator_name,
            'tables': []
        }
        
        # Process each table configuration from approach config
        for table_config in self._config.tables:
            table_dict = OmegaConf.to_container(table_config, resolve=True)
            
            # Handle data assignment based on table name
            if table_config.name == 'subjects':
                # Extract unique subjects
                subject_col = self._config.data.subject_column
                subjects_df = data[[subject_col]].drop_duplicates().reset_index(drop=True)
                table_dict['data'] = subjects_df
            else:
                # Use full dataset
                table_dict['data'] = data
            
            # Add privacy settings to tabular model config if enabled
            if self._config.privacy.enabled and 'tabular_model_configuration' in table_dict:
                table_dict['tabular_model_configuration']['differential_privacy'] = {
                    'max_epsilon': self._config.privacy.max_epsilon,
                    'delta': self._config.privacy.delta,
                }
            
            mostly_config['tables'].append(table_dict)
        
        return mostly_config
    
    def train_and_evaluate(self, data: pd.DataFrame) -> dict[str, float]:
        """Train generator and evaluate it.
        
        Args:
            data: Training data
            
        Returns:
            Evaluation metrics from MOSTLY AI QA report
        """
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "dataset_rows": data.shape[0],
                "dataset_cols": data.shape[1],
                "approach": self._config.approach,
                "subject_column": self._config.data.subject_column,
                "max_training_time": self._config.training.max_training_time,
                "max_epochs": self._config.training.max_epochs,
                "privacy_enabled": self._config.privacy.enabled,
                "generator_name": self._config.generator_name,
            })
            
            # Log data analysis
            subject_col = self._config.data.subject_column
            num_subjects = data[subject_col].nunique()
            avg_sequence_length = len(data) / num_subjects
            
            logger.info(f"📊 Dataset analysis:")
            logger.info(f"   • Total rows: {data.shape[0]:,}")
            logger.info(f"   • Total columns: {data.shape[1]}")
            logger.info(f"   • Unique subjects: {num_subjects:,}")
            logger.info(f"   • Avg sequence length: {avg_sequence_length:.1f}")
            logger.info(f"🔧 Using {self._config.approach} approach")
            
            mlflow.log_metrics({
                "num_subjects": num_subjects,
                "avg_sequence_length": avg_sequence_length,
            })
            
            # Build MOSTLY AI configuration
            mostly_config = self._build_mostly_config(data)
            
            # Train generator
            logger.info(f"🚀 Training generator: {self._config.generator_name}")
            
            start_time = time.time()
            generator = self._mostly.train(
                config=mostly_config,
                start=True,
                wait=True
            )
            training_time = time.time() - start_time
            
            mlflow.log_metric("training_time_seconds", training_time)
            logger.info(f"✅ Training completed in {training_time:.1f}s")
            
            # Get MOSTLY AI's built-in quality metrics
            logger.info("🔍 Getting MOSTLY AI quality report...")
            generator.reports(display=True)
    


__all__ = ["Trainer"]
