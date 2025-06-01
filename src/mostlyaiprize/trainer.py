"""Simplified trainer for MOSTLY AI Prize Challenge 2."""

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mostlyai.sdk import MostlyAI
from omegaconf import DictConfig, OmegaConf

from mostlyaiprize.report_parser import parse_report_metrics

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
          
            mostly_config['tables'].append(table_dict)
        
        return mostly_config
    
    def _extract_and_log_metrics(self, generator: Any) -> dict[str, float]:
        """Extract metrics from MOSTLY AI report and log to MLflow.
        
        Args:
            generator: Trained MOSTLY AI generator
            
        Returns:
            Extracted quality metrics
        """
        try:
            # Get the QA report as HTML string
            # Note: This might need adjustment based on actual MOSTLY AI SDK API
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
                # Save report to temporary file (adjust this based on actual SDK API)
                generator.qa_report.save(tmp_file.name)
                tmp_path = Path(tmp_file.name)
            
            # Read and parse the HTML report
            html_content: str = tmp_path.read_text(encoding='utf-8')
            
            # Log the full HTML report as MLflow artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as report_file:
                report_file.write(html_content)
                report_path = Path(report_file.name)
            
            mlflow.log_artifact(str(report_path), "reports")
            logger.info("📄 Full HTML report logged to MLflow")
            
            # Extract and log quality metrics
            metrics = parse_report_metrics(html_content)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            logger.info("📊 Quality metrics extracted and logged:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"   • {metric_name}: {metric_value}")
            
            # Clean up temporary files
            tmp_path.unlink()
            report_path.unlink()
            
            return metrics
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to extract quality metrics: {e}")
            return {}
    
    def train_and_evaluate(self, data: pd.DataFrame) -> dict[str, float]:
        """Train generator and evaluate it.
        
        Args:
            data: Training data
            
        Returns:
            Evaluation metrics from MOSTLY AI QA report
        """
        with mlflow.start_run():
            # Log parameters
            mlflow.log_dict(dict(self._config), "config.json")
            
            # Build MOSTLY AI configuration
            mostly_config = self._build_mostly_config(data)
            breakpoint()
            
            # Train generator
            logger.info(f"🚀 Training generator: {self._config.generator_name}")
            
            start_time: float = time.time()
            generator = self._mostly.train(
                config=mostly_config,
                start=True,
                wait=True
            )
            training_time: float = time.time() - start_time
            
            mlflow.log_metric("training_time_seconds", training_time)
            logger.info(f"✅ Training completed in {training_time:.1f}s")
            
            # Extract and log quality metrics
            quality_metrics = self._extract_and_log_metrics(generator)
            
            # Display built-in MOSTLY AI reports
            logger.info("🔍 Displaying MOSTLY AI quality report...")
            generator.reports(display=True)
            
            return quality_metrics


__all__ = ["Trainer"]
