#!/usr/bin/env python3
"""Generate final competition submissions."""

import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mostlyaiprize.core.config import CompetitionConfig
from mostlyaiprize.core.trainer import CompetitionTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_generator(trainer: CompetitionTrainer, generator_path: Path, generator_name: str):
    """Load a trained generator."""
    try:
        if generator_path.exists():
            logger.info(f"📥 Loading generator from: {generator_path}")
            generator = trainer.mostly.generators.import_from_file(str(generator_path))
            trainer.generators[generator_name] = generator
            return generator
        else:
            logger.error(f"❌ Generator not found: {generator_path}")
            return None
    except Exception as e:
        logger.error(f"❌ Failed to load generator: {e}")
        return None


def generate_challenge_submission(
    trainer: CompetitionTrainer, 
    generator_name: str, 
    original_data: pd.DataFrame,
    challenge_name: str
) -> Path:
    """Generate submission for a specific challenge."""
    logger.info(f"🎯 Generating {challenge_name} submission...")
    
    # Target size = original dataset size
    target_size = len(original_data)
    
    try:
        # Generate synthetic data
        start_time = time.time()
        synthetic_data = trainer.generate_synthetic_data(
            generator_name, 
            size=target_size,
            seed=42  # For reproducibility
        )
        generation_time = time.time() - start_time
        
        logger.info(f"✅ Generated {len(synthetic_data):,} samples in {generation_time:.1f}s")
        
        # Validate submission format
        if synthetic_data.shape[1] != original_data.shape[1]:
            logger.error(f"❌ Column mismatch! Expected {original_data.shape[1]}, got {synthetic_data.shape[1]}")
            return None
        
        if not all(col in synthetic_data.columns for col in original_data.columns):
            logger.error("❌ Column names don't match original dataset!")
            return None
        
        # Reorder columns to match original
        synthetic_data = synthetic_data[original_data.columns]
        
        # Save submission
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = Path("submissions") / f"{challenge_name}_submission_{timestamp}.csv"
        submission_path.parent.mkdir(exist_ok=True)
        
        synthetic_data.to_csv(submission_path, index=False)
        logger.info(f"💾 Submission saved: {submission_path}")
        
        # Quick quality check
        logger.info("🔍 Quick quality check:")
        logger.info(f"  Shape: {synthetic_data.shape} (expected: {original_data.shape})")
        logger.info(f"  Memory: {synthetic_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        logger.info(f"  Null values: {synthetic_data.isnull().sum().sum()}")
        
        # Check for potential issues
        if synthetic_data.isnull().all().any():
            logger.warning("⚠️ Some columns are entirely null!")
        
        if (synthetic_data == synthetic_data.iloc[0]).all().any():
            logger.warning("⚠️ Some columns have constant values!")
        
        return submission_path
        
    except Exception as e:
        logger.error(f"❌ Failed to generate {challenge_name} submission: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Generate competition submissions."""
    logger.info("🏆 Generating MOSTLY AI Prize submissions...")
    
    # Initialize trainer
    config = CompetitionConfig()
    trainer = CompetitionTrainer(config)
    
    submissions_generated = []
    
    # Challenge 1 submission
    challenge1_generator_path = Path("generators/challenge1_generator.zip")
    if challenge1_generator_path.exists():
        logger.info("📊 Processing Challenge 1...")
        
        # Load generator
        generator = load_generator(trainer, challenge1_generator_path, "challenge1_generator")
        if generator:
            # Find original dataset
            data_dir = Path("data")
            possible_files = ["challenge1_train.csv", "challenge1.csv", "dataset1.csv"]
            
            dataset_path = None
            for filename in possible_files:
                candidate = data_dir / filename
                if candidate.exists():
                    dataset_path = candidate
                    break
            
            if dataset_path is None:
                csv_files = list(data_dir.glob("*.csv"))
                if csv_files:
                    dataset_path = csv_files[0]
            
            if dataset_path:
                try:
                    original_df = pd.read_csv(dataset_path)
                    submission_path = generate_challenge_submission(
                        trainer, "challenge1_generator", original_df, "challenge1"
                    )
                    if submission_path:
                        submissions_generated.append(submission_path)
                except Exception as e:
                    logger.error(f"❌ Failed to process Challenge 1: {e}")
            else:
                logger.error("❌ Challenge 1 dataset not found!")
    else:
        logger.warning("⚠️ Challenge 1 generator not found. Run training first.")
    
    # Challenge 2 submission
    challenge2_generator_path = Path("generators/challenge2_generator.zip")
    if challenge2_generator_path.exists():
        logger.info("📊 Processing Challenge 2...")
        
        # Load generator
        generator = load_generator(trainer, challenge2_generator_path, "challenge2_generator")
        if generator:
            # Find original dataset
            data_dir = Path("data")
            possible_files = ["challenge2_train.csv", "challenge2.csv", "dataset2.csv"]
            
            dataset_path = None
            for filename in possible_files:
                candidate = data_dir / filename
                if candidate.exists():
                    dataset_path = candidate
                    break
            
            if dataset_path is None:
                csv_files = list(data_dir.glob("*.csv"))
                if len(csv_files) >= 2:
                    dataset_path = csv_files[1]
            
            if dataset_path:
                try:
                    original_df = pd.read_csv(dataset_path)
                    submission_path = generate_challenge_submission(
                        trainer, "challenge2_generator", original_df, "challenge2"
                    )
                    if submission_path:
                        submissions_generated.append(submission_path)
                except Exception as e:
                    logger.error(f"❌ Failed to process Challenge 2: {e}")
            else:
                logger.error("❌ Challenge 2 dataset not found!")
    else:
        logger.warning("⚠️ Challenge 2 generator not found. Run training first.")
    
    # Summary
    if submissions_generated:
        logger.info("🎉 Submission generation completed!")
        logger.info("📂 Generated submissions:")
        for submission in submissions_generated:
            logger.info(f"  {submission}")
        
        logger.info("\n📋 Next steps:")
        logger.info("1. Run validation: hatch run dev:validate-submission")
        logger.info("2. Upload to competition platform")
        logger.info("3. Submit before deadline!")
        
        # Create submission summary
        summary_path = Path("submissions") / f"submission_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(summary_path, 'w') as f:
            f.write("# MOSTLY AI Prize Submission Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Files Generated:\n\n")
            for submission in submissions_generated:
                f.write(f"- `{submission.name}`\n")
            f.write("\n## Next Steps:\n\n")
            f.write("1. Validate submissions\n")
            f.write("2. Upload to competition platform\n")
            f.write("3. Submit before deadline\n")
        
        logger.info(f"📋 Summary saved: {summary_path}")
        
    else:
        logger.error("❌ No submissions generated! Please train generators first.")
        logger.info("💡 Try running:")
        logger.info("  hatch run dev:train-challenge1")
        logger.info("  hatch run dev:train-challenge2")
        sys.exit(1)


if __name__ == "__main__":
    main()
