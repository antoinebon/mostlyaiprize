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
    # Load data
    data_file = Path(cfg.data.path)
    if not data_file.exists():
        logger.error(f"❌ Data file not found: {data_file}")
        return
    
    logger.info(f"📖 Loading dataset: {data_file}")
    df = pd.read_csv(data_file)
    logger.info(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
  
    # Train and evaluate
    trainer = Trainer(cfg)
    trainer.train_and_evaluate(df)
      
    logger.info("🎉 Challenge 2 training completed!")


if __name__ == "__main__":
    main()
