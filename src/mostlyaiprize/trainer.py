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
from flatten_dict import flatten

from .report_parser import ReportParser
from .features import SubjectTableEngineer

logger = logging.getLogger(__name__)

def flatten_nested(obj, parent_key='', sep='.'):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_nested(v, new_key, sep=sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(flatten_nested(v, new_key, sep=sep))
    else:
        items[parent_key] = obj
    return items



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
        mostly_config = {"name": self._config.generator_name, "tables": []}

        # Process each table configuration from approach config
        for table_config in self._config.tables:
            table_dict = OmegaConf.to_container(table_config, resolve=True)

            # Handle data assignment based on table name
            if table_config.name == "subjects":
                # Extract unique subjects
                t0 = time.time()
                subjects_df = SubjectTableEngineer(self._config.data.subject_column).create_enhanced_subject_table(data)
                t1 = time.time()-t0
                print(t1)
                table_dict["data"] = subjects_df
            else:
                # Use full dataset
                table_dict["data"] = data

            mostly_config["tables"].append(table_dict)

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

            # breakpoint()
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_zip_path = Path(temp_dir) / "quality_report.zip"
                generator.reports(file_path=str(temp_zip_path), display=False)

                logger.info(f"📄 Report generated at: {temp_zip_path}")

                # Extract HTML content from ZIP
                with zipfile.ZipFile(temp_zip_path, "r") as zip_file:
                    # Find HTML files in the ZIP
                    for html_file_name in zip_file.namelist():
                        html_content = zip_file.read(html_file_name).decode("utf-8")

                        # Save and log the unzipped HTML report
                        html_file_path = Path(temp_dir) / html_file_name
                        html_file_path.write_text(html_content)

                        # Log the HTML file with a descriptive name
                        mlflow.log_artifact(str(html_file_path))
                        logger.info(f"📄 {html_file_name} logged to MLflow")

                        # Parse metrics from HTML content
                        parser = ReportParser(html_content)
                        metrics = parser.extract_metrics()

                        # Log metrics to MLflow
                        for metric_name, metric_value in metrics.items():
                            full_metric_name = "_".join((html_file_name.split('-')[0], metric_name))
                            mlflow.log_metric(full_metric_name, metric_value)
                            logger.info(f"   • {full_metric_name}: {metric_value}")


        except Exception as e:
            logger.warning(f"⚠️  Failed to extract quality metrics: {e}")

    def train_and_evaluate(self, data: pd.DataFrame):
        """Train generator and evaluate it.

        Args:
            data: Training data

        Returns:
            Evaluation metrics from MOSTLY AI QA report
        """
        with mlflow.start_run():

            # Log config
            config = OmegaConf.to_container(self._config, resolve=True)
            mlflow.log_dict(config, "config.json")
                
            # Build MOSTLY AI configuration
            mostly_config = self._build_mostly_config(data)

            # Log key parameters
            for table in mostly_config["tables"]:
                for key, value in flatten(table["tabular_model_configuration"], reducer="underscore").items():
                    mlflow.log_param(f"{table['name']}_{key}", value)

            # Train generator
            logger.info(f"🚀 Training generator: {self._config.generator_name}")

            start_time: float = time.time()
            generator = self._mostly.train(config=mostly_config, start=True, wait=True)
            training_time: float = time.time() - start_time

            mlflow.log_metric("training_time_seconds", training_time)
            logger.info(f"✅ Training completed in {training_time:.1f}s")

            # Extract and log quality metrics
            self.extract_and_log_metrics(generator)

            # Display built-in MOSTLY AI reports in UI (if desired)
            logger.info("🔍 Quality report generated and metrics logged to MLflow")

            # Create submission
            sd = self._mostly.generate(generator)
            syn = sd.data()
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "generated_sequences.csv.gz"
                syn["sequences"].to_csv(temp_path, index=False)
                mlflow.log_artifact(temp_path)


__all__ = ["Trainer"]
