"""Simple trainer for MOSTLY AI Prize Challenge 2."""

import logging
import time
from typing import Any

import mlflow
import pandas as pd
from mostlyai.sdk import MostlyAI
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Simple trainer for Challenge 2."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize trainer with configuration.
        
        Args:
            config: Configuration object
        """
        self._config: DictConfig = config
        self._mostly: MostlyAI = MostlyAI(local=True)
        
        # Setup MLflow
        mlflow.set_experiment(config.experiment_name)
    
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
                "privacy_level": self._config.privacy_level,
                "random_seed": self._config.random_seed,
                "generator_name": self._config.generator_name,
            })
            
            # Train generator
            logger.info(f"🚀 Training generator: {self._config.generator_name}")
            logger.info(f"📊 Dataset: {data.shape[0]:,} rows × {data.shape[1]} columns")
            
            start_time = time.time()
            generator = self._mostly.train(
                name=self._config.generator_name,
                data=data
            )
            training_time = time.time() - start_time
            
            mlflow.log_metric("training_time_seconds", training_time)
            logger.info(f"✅ Training completed in {training_time:.1f}s")
            
            # Get MOSTLY AI's built-in quality metrics
            logger.info("🔍 Getting MOSTLY AI quality report...")
            metrics = generator.reports(display=True)
            breakpoint()
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log results
            logger.info("📊 Challenge 2 Results:")
            logger.info(metrics)
            
            return metrics


__all__ = ["Trainer"]
