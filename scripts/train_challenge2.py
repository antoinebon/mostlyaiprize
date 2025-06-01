"""TODO:
* use mlflow for experiment tracking, log artifacts instead of saving to disk
* simplify config. use hydra for config management
"""
"""Train synthetic data generator for Challenge 2."""

import logging
import os
import sys
import time
from pathlib import Path
import json

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mostlyaiprize.core.config import CompetitionConfig
from mostlyaiprize.core.trainer import CompetitionTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/challenge2_training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Train Challenge 2 generator."""
    logger.info("🚀 Starting Challenge 2 training...")
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Find Challenge 2 dataset
    data_dir = Path("data")
    dataset_path = data_dir/ "equential-training.csv"
    
    
    # Load dataset
    logger.info(f"📖 Loading Challenge 2 dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    logger.info(f"📋 Column types: {dict(df.dtypes.value_counts())}")
    
    # Configuration for Challenge 2 (might need different settings)
    config = CompetitionConfig(
        privacy_level=os.getenv("PRIVACY_LEVEL", "medium"),
        time_budget_hours=float(os.getenv("TIME_BUDGET_HOURS", "6.0")),
        random_seed=int(os.getenv("RANDOM_SEED", "42"))
    )
    
    # Adjust config for Challenge 2 if needed
    # (You might want to tune these based on Challenge 2 characteristics)
    
    trainer = CompetitionTrainer(config)
    
    # Train generator
    generator_name = "challenge2_generator"
    logger.info(f"🎯 Training generator: {generator_name}")
    
    start_time = time.time()
    try:
        generator = trainer.train_generator(df, generator_name)
        training_time = time.time() - start_time
        
        logger.info(f"✅ Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
        # Evaluation
        logger.info("🔍 Running evaluation...")
        evaluation = trainer.evaluate_generator(generator_name, df, test_size=min(2000, len(df)//2))
        
        logger.info("📊 Challenge 2 Results:")
        for metric, value in evaluation.items():
            logger.info(f"  {metric}: {value}")
        
        # Save artifacts
        export_path = f"generators/{generator_name}.zip"
        Path("generators").mkdir(exist_ok=True)
        generator.export_to_file(export_path)
        logger.info(f"💾 Generator saved to: {export_path}")
        
        # Save results
        results_path = f"results/challenge2_evaluation.json"
        Path("results").mkdir(exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                'dataset_info': {
                    'shape': df.shape,
                    'file': str(dataset_path),
                    'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
                },
                'training_info': {
                    'training_time_seconds': training_time,
                    'generator_name': generator_name,
                    'config': config.__dict__
                },
                'evaluation': evaluation
            }, f, indent=2, default=str)
        
        logger.info(f"📋 Results saved to: {results_path}")
        logger.info("🎉 Challenge 2 training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
