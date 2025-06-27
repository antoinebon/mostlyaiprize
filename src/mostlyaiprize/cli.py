"""Enhanced CLI for MOSTLY AI Prize supporting both challenges."""

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from mostlyaiprize.trainer import Trainer

level = logging.DEBUG
logging.basicConfig(level=level)
logging.getLogger("mostlyai").setLevel(level)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="base_config", version_base=None)
def main(config: DictConfig) -> None:
    """Train MOSTLY AI Prize generators with enhanced configuration support.

    Supports both sequential data (Challenge 2) and flat/tabular data challenges.

    Examples:

        FLAT DATA CHALLENGE:
        # Train flat data generator
        python -m mostlyaiprize.cli --config-name=config_flat

        # Override settings
        python -m mostlyaiprize.cli --config-name=config_flat data.path="my-flat-data.csv"
        python -m mostlyaiprize.cli --config-name=config_flat random_seed=123

        SEQUENTIAL DATA CHALLENGE:
        # Train sequential data generator
        python -m mostlyaiprize.cli --config-name=config_sequential

        # Override settings
        python -m mostlyaiprize.cli --config-name=config_sequential data.path="my-sequential-data.csv"
        python -m mostlyaiprize.cli --config-name=config_sequential feature_engineering.enable_entropy=true

        GENERAL OVERRIDES:
        # Different MLflow experiment
        python -m mostlyaiprize.cli --config-name=config_flat mlflow.experiment_name="my_experiment"

        # Remote MLflow tracking
        python -m mostlyaiprize.cli --config-name=config_flat mlflow.tracking_uri="http://localhost:5000"

        # Different random seed
        python -m mostlyaiprize.cli --config-name=config_flat random_seed=999

        GENERAL:
        # Configure MLflow tracking
        python -m mostlyaiprize.cli mlflow.tracking_uri="http://localhost:5000" mlflow.experiment_name="my_experiment"

        # Production run with remote MLflow
        python -m mostlyaiprize.cli --config-name=config_production mlflow.tracking_uri="https://mlflow.example.com"

        # Custom data path
        python -m mostlyaiprize.cli data.path="path/to/my/data.csv"

        # Different random seed
        python -m mostlyaiprize.cli random_seed=123
    """
    # Get challenge type from config
    challenge_type = config.challenge_type

    # Log random seed if specified
    if config.get("random_seed"):
        logger.info(f"🎲 Using random seed: {config.random_seed}")

    # Load data
    data_file = Path(config.data.path)
    if not data_file.exists():
        logger.error(f"❌ Data file not found: {data_file}")
        return

    logger.info(f"📖 Loading dataset: {data_file.name}")
    df: pd.DataFrame = pd.read_csv(data_file)
    logger.info(f"✅ Dataset loaded: {df.shape[0]:,} rows")

    # Train and evaluate
    trainer = Trainer(config)
    trainer.train_and_evaluate(df)

    logger.info(f"🎉 {challenge_type.title()} challenge training completed!")


if __name__ == "__main__":
    main()
