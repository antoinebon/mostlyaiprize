"""Enhanced trainer for MOSTLY AI Prize supporting both sequential and flat data challenges."""

import logging
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any
import random

from omegaconf import DictConfig, OmegaConf
import torch
import mlflow
import pandas as pd
import numpy as np
from mostlyai.sdk import MostlyAI
from mostlyai.sdk.domain import (
    GeneratorConfig,
    ModelConfiguration,
)

from .report_parser import ReportParser
from .features import SubjectTableEngineer

logger = logging.getLogger(__name__)


class Trainer:
    """Enhanced trainer supporting both sequential and flat data challenges."""

    def __init__(self, config: DictConfig) -> None:
        """Initialize trainer with configuration.

        Args:
            config: Configuration object
        """
        self._config: DictConfig = config
        self._mostly: MostlyAI = MostlyAI(local=True)
        self._challenge_type: str = self._config.challenge_type

        # Setup MLflow with configuration
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking with settings from configuration."""
        # Handle both old and new config formats
        if hasattr(self._config, "mlflow"):
            mlflow_config = self._config.mlflow

            # Set tracking URI if specified
            if hasattr(mlflow_config, "tracking_uri") and mlflow_config.tracking_uri is not None:
                mlflow.set_tracking_uri(mlflow_config.tracking_uri)
                logger.info(f"📊 MLflow tracking URI: {mlflow_config.tracking_uri}")
            else:
                logger.info("📊 MLflow tracking URI: ./mlruns (default local)")

            # Enable system metrics if requested
            if hasattr(mlflow_config, "enable_system_metrics") and mlflow_config.enable_system_metrics:
                mlflow.enable_system_metrics_logging()
                logger.info("📈 MLflow system metrics logging enabled")

            # Set experiment
            experiment_name = mlflow_config.experiment_name
        else:
            # Fallback to old config format
            experiment_name = self._config.experiment_name
            logger.info("📊 MLflow tracking URI: ./mlruns (default local)")

        mlflow.set_experiment(experiment_name)
        logger.info(f"🔬 MLflow experiment: {experiment_name}")

    def _build_mostly_config(self, data: pd.DataFrame) -> dict[str, Any]:
        """Build MOSTLY AI configuration from Hydra config and data.

        Args:
            data: Training data

        Returns:
            MOSTLY AI configuration dict
        """
        # Start with base generator config
        mostly_config = {"name": self._config.generator_name, "tables": []}

        # Add random seed if available (check if MOSTLY AI supports it)
        if self._config.get("random_seed"):
            mostly_config["random_state"] = self._config.random_seed
            logger.info(f"🎲 Adding random seed to MOSTLY AI config: {self._config.random_seed}")

        # Process each table configuration from approach config
        for table_config in self._config.tables:
            table_dict = OmegaConf.to_container(table_config, resolve=True)

            # Handle sequential data tables
            if self._challenge_type == "sequential" and  table_config.name == "subjects":
                # Create engineer with configuration-driven parameters
                feature_config =self._config.get("subjects_feature_engineering") 
                if feature_config:
                    logger.info("📊 Generating additional features for Subject")
                    engineer = SubjectTableEngineer(
                        subject_column=self._config.data.subject_column,
                        enable_sequential=feature_config.get("enable_sequential", True),
                        enable_distribution=feature_config.get("enable_distribution", True),
                        enable_cross_column=feature_config.get("enable_cross_column", True),
                        enable_entropy=feature_config.get("enable_entropy", False),
                        enable_correlations=feature_config.get("enable_correlations", True),
                    )
                    table_dict["data"] = engineer.create_enhanced_subject_table(data)
                    logger.info(f"📊 Subject table engineering completed - {data.shape[1]} features created")
                else:
                    table_dict["data"] = data[[self._config.data.subject_column]].drop_duplicates()
            else:
                table_dict["data"] = data

            logger.info(f"📊 {table_config.name} table shape: {table_dict['data'].shape}")

            if table_dict.get("tabular_model_configuration") is not None:
                table_dict["tabular_model_configuration"] = ModelConfiguration(
                    **table_dict["tabular_model_configuration"]
                )

            mostly_config["tables"].append(table_dict)

        return GeneratorConfig(**mostly_config)

    def extract_and_log_metrics(self, generator: Any) -> None:
        """Extract metrics from MOSTLY AI report and log to MLflow.

        Args:
            generator: Trained MOSTLY AI generator
        """
        try:
            # Generate the QA report (provide temporary path for ZIP file)
            logger.info("📋 Generating quality report...")

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_zip_path: Path = Path(temp_dir) / "quality_report.zip"
                generator.reports(file_path=str(temp_zip_path), display=False)

                logger.info(f"📄 Report generated at: {temp_zip_path}")

                # Extract HTML content from ZIP
                with zipfile.ZipFile(temp_zip_path, "r") as zip_file:
                    # Find HTML files in the ZIP
                    for html_file_name in zip_file.namelist():
                        html_content: str = zip_file.read(html_file_name).decode("utf-8")

                        # Save and log the unzipped HTML report
                        html_file_path: Path = Path(temp_dir) / html_file_name
                        html_file_path.write_text(html_content)

                        # Log the HTML file with a descriptive name
                        mlflow.log_artifact(str(html_file_path))
                        logger.info(f"📄 {html_file_name} logged to MLflow")

                        # Parse metrics from HTML content
                        parser = ReportParser(html_content)
                        metrics: dict[str, float] = parser.extract_metrics()

                        # Log metrics to MLflow
                        for metric_name, metric_value in metrics.items():
                            full_metric_name: str = "_".join((html_file_name.split("-")[0], metric_name))
                            mlflow.log_metric(full_metric_name, metric_value)
                            logger.info(f"   • {full_metric_name}: {metric_value}")

        except Exception as e:
            logger.warning(f"⚠️  Failed to extract quality metrics: {e}")

    def train_and_evaluate(self, data: pd.DataFrame) -> None:
        """Train generator and evaluate it.

        Args:
            data: Training data
        """
        with mlflow.start_run():
            # Log challenge type and random seed
            mlflow.log_param("challenge_type", self._challenge_type)
            mlflow.log_param("random_seed", self._config.get("random_seed"))

            logger.info(f"🎯 Challenge type: {self._challenge_type}")

            # Log config
            config = OmegaConf.to_container(self._config, resolve=True)
            mlflow.log_dict(config, "config.json")

            # Build MOSTLY AI configuration
            mostly_config = self._build_mostly_config(data)

            # Log key parameters
            for table in mostly_config.tables:
                for key, value in table.tabular_model_configuration.model_dump().items():
                    mlflow.log_param(f"{table.name}_{key}", value)

            # Log challenge-specific parameters
            if self._challenge_type == "sequential":
                for param_name, param_value in config.get("subjects_feature_engineering", {}).items():
                    mlflow.log_param(f"feature_engineering_{param_name}", param_value)
                    logger.info(f"📈 Feature engineering - {param_name}: {param_value}")

            # Log MLflow configuration parameters
            for param_name, param_value in config.get("mlflow", {}).items():
                mlflow.log_param(f"mlflow_{param_name}", param_value)

            # Train generator
            logger.info(f"🚀 Training generator: {self._config.generator_name}")

            start_time = time.time()

            generator = self._mostly.train(config=mostly_config, start=True, wait=True, progress_bar=True)
            mlflow.log_param("generator_id", generator.id)

            training_time = time.time() - start_time

            mlflow.log_metric("training_time_seconds", training_time)
            logger.info(f"✅ Training completed in {training_time:.1f}s")

            # Extract and log quality metrics
            self.extract_and_log_metrics(generator)

            # Display built-in MOSTLY AI reports in UI (if desired)
            logger.info("🔍 Quality report generated and metrics logged to MLflow")

            if self._config.get("generate_data", True):
                # Create submission
                syn = self._mostly.generate(generator).data()

                with tempfile.TemporaryDirectory() as temp_dir:
                    if self._challenge_type == "sequential":
                        # Sequential data submission
                        temp_path = Path(temp_dir) / "generated_sequences.csv.gz"
                        syn["sequences"].to_csv(temp_path, index=False)
                    else:
                        # Flat data submission
                        temp_path = Path(temp_dir) / "generated_flat_data.csv.gz"
                        syn.to_csv(temp_path, index=False)

                    mlflow.log_artifact(temp_path)
                    logger.info(f"💾 Generated data saved: {temp_path.name}")


__all__ = ["Trainer"]
