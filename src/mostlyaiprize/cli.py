"""Enhanced CLI for MOSTLY AI Prize Challenge 2."""

import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from mostlyaiprize.trainer import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train Challenge 2 generator with enhanced configuration support.
    
    Examples:
        # Debug mode (very fast training)
        python -m mostlyaiprize.cli --config-name=config_debug
        
        # Single table approach with fast training
        python -m mostlyaiprize.cli --config-name=config_single_table_fast
        
        # Multi table approach with fast training
        python -m mostlyaiprize.cli --config-name=config_multi_table_fast
        
        # Override specific settings
        python -m mostlyaiprize.cli approach=multi_table training=debug
        
        # Override privacy settings
        python -m mostlyaiprize.cli privacy.enabled=false privacy.max_epsilon=10.0
        
        # Production run
        python -m mostlyaiprize.cli --config-name=config_production
    """
    logger.info("🚀 Starting Challenge 2 training...")
    logger.info(f"🔧 Configuration:")
    logger.info(f"   • Approach: {cfg.approach}")
    logger.info(f"   • Training time: {cfg.training.max_training_time}")
    logger.info(f"   • Max epochs: {cfg.training.max_epochs}")
    logger.info(f"   • Privacy: {'enabled' if cfg.privacy.enabled else 'disabled'}")
    if cfg.privacy.enabled:
        logger.info(f"   • Privacy epsilon: {cfg.privacy.max_epsilon}")
    logger.info(f"   • Subject column: {cfg.data.subject_column}")
    
    # Load data
    data_file = Path(cfg.data.path)
    if not data_file.exists():
        logger.error(f"❌ Data file not found: {data_file}")
        return
    
    logger.info(f"📖 Loading dataset: {data_file}")
    df = pd.read_csv(data_file)
    logger.info(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Validate subject column exists
    if cfg.data.subject_column not in df.columns:
        logger.error(f"❌ Subject column '{cfg.data.subject_column}' not found in data")
        logger.info(f"Available columns: {list(df.columns)}")
        return
    
    # Train and evaluate
    trainer = Trainer(cfg)
    trainer.train_and_evaluate(df)
      
    logger.info("🎉 Challenge 2 training completed!")


if __name__ == "__main__":
    main()
