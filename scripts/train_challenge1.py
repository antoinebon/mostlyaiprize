#!/usr/bin/env python3
"""Train synthetic data generator for Challenge 1."""

import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from mostlyai.sdk import MostlyAI

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
        logging.FileHandler('logs/challenge1_training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Train Challenge 1 generator."""
    logger.info("🚀 Starting Challenge 1 training...")
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Find Challenge 1 dataset
    data_dir = Path("data")
    possible_files = [
        "challenge1_train.csv",
        "challenge1.csv", 
        "train_challenge1.csv",
        "dataset1.csv"
    ]
    
    dataset_path = None
    for filename in possible_files:
        candidate = data_dir / filename
        if candidate.exists():
            dataset_path = candidate
            break
    
    if dataset_path is None:
        # Look for any CSV files in data directory
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            logger.info(f"📂 Found CSV files: {[f.name for f in csv_files]}")
            dataset_path = csv_files[0]  # Use first one
            logger.info(f"📊 Using dataset: {dataset_path.name}")
        else:
            logger.error("❌ No dataset found! Please run: hatch run dev:download-data")
            sys.exit(1)
    
    # Load dataset
    logger.info(f"📖 Loading dataset: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        # Basic dataset info
        logger.info(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        logger.info(f"📋 Column types: {dict(df.dtypes.value_counts())}")
        
        missing_info = df.isnull().sum()
        if missing_info.sum() > 0:
            logger.info(f"🔍 Missing values: {dict(missing_info[missing_info > 0])}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        sys.exit(1)
    
    # Initialize configuration and trainer
    config = CompetitionConfig(
        privacy_level=os.getenv("PRIVACY_LEVEL", "medium"),
        time_budget_hours=float(os.getenv("TIME_BUDGET_HOURS", "6.0")),
        random_seed=int(os.getenv("RANDOM_SEED", "42"))
    )
    
    trainer = CompetitionTrainer(config)
    
    # Train generator
    generator_name = "challenge1_generator"
    logger.info(f"🎯 Training generator: {generator_name}")
    
    start_time = time.time()
    try:
        generator = trainer.train_generator(df, generator_name)
        training_time = time.time() - start_time
        
        logger.info(f"✅ Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        
        # Generate quality report
        logger.info("📈 Generating quality report...")
        try:
            reports = generator.reports(display=False)
            logger.info("✅ Quality report generated")
        except Exception as e:
            logger.warning(f"⚠️ Could not generate quality report: {e}")
        
        # Quick evaluation
        logger.info("🔍 Running evaluation...")
        evaluation = trainer.evaluate_generator(generator_name, df, test_size=min(2000, len(df)//2))
        
        logger.info("📊 Challenge 1 Results:")
        for metric, value in evaluation.items():
            logger.info(f"  {metric}: {value}")
        
        # Save generator
        export_path = f"generators/{generator_name}.zip"
        Path("generators").mkdir(exist_ok=True)
        generator.export_to_file(export_path)
        logger.info(f"💾 Generator saved to: {export_path}")
        
        # Save evaluation results
        results_path = f"results/challenge1_evaluation.json"
        Path("results").mkdir(exist_ok=True)
        
        import json
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
        logger.info("🎉 Challenge 1 training completed successfully!")
        
        # Competition score summary
        comp_score = evaluation.get('competition_score', 0.0)
        if comp_score > 0.8:
            logger.info("🏆 Excellent performance! Competition ready.")
        elif comp_score > 0.6:
            logger.info("👍 Good performance! Consider fine-tuning.")
        else:
            logger.info("⚠️ Consider adjusting hyperparameters or privacy settings.")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
