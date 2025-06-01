"""Simplified trainer for MOSTLY AI Prize Challenge 2."""

import logging
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mostlyai.sdk import MostlyAI
from omegaconf import DictConfig, OmegaConf

from mostlyaiprize.report_parser import ReportParser

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
    
    def extract_and_log_metrics(self, generator: Any) -> None:
        """Extract metrics from MOSTLY AI report and log to MLflow.
        
        Args:
            generator: Trained MOSTLY AI generator
            
        Returns:
            Extracted quality metrics
        """
        try:
            # Generate the QA report (provide temporary path for ZIP file)
            logger.info("📋 Generating quality report...")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_zip_path = Path(temp_dir) / "quality_report.zip"
                generator.reports(file_path=str(temp_zip_path), display=False)
                
                logger.info(f"📄 Report generated at: {temp_zip_path}")
                
                # Extract HTML content from ZIP
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_file:
                    # Find HTML files in the ZIP
                    html_file_name = zip_file.namelist()[0]
                    html_content = zip_file.read(html_file_name).decode('utf-8')
                
                # Save and log the unzipped HTML report
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as html_file:
                    html_file.write(html_content)
                    html_temp_path = Path(html_file.name)
                
                # Log the HTML file with a descriptive name
                mlflow.log_artifact(str(html_file_name))
                logger.info("📄 HTML report logged to MLflow")
                
                # Parse metrics from HTML content
                parser = ReportParser(html_content)
                metrics = parser.extract_metrics()
                
                # Log metrics to MLflow
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                logger.info("📊 Quality metrics extracted and logged:")
                for metric_name, metric_value in metrics.items():
                    logger.info(f"   • {metric_name}: {metric_value}")
                
                # Clean up temporary HTML file
                html_temp_path.unlink()
                
            
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
            quality_metrics = self.extract_and_log_metrics(generator)
            
            # Display built-in MOSTLY AI reports in UI (if desired)
            logger.info("🔍 Quality report generated and metrics logged to MLflow")
            
            return quality_metrics


__all__ = ["Trainer"]
