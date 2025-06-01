"""Core training functionality for the competition."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from mostlyai.sdk import MostlyAI

from mostlyaiprize.core.config import CompetitionConfig
from mostlyaiprize.evaluation.metrics import PrivacyUtilityEvaluator

logger = logging.getLogger(__name__)


class CompetitionTrainer:
    """Main trainer class for the MOSTLY AI Prize competition."""
    
    def __init__(self, config: CompetitionConfig) -> None:
        """Initialize trainer with configuration."""
        self.config = config
        self.mostly = MostlyAI(local=config.local_mode)
        self.evaluator = PrivacyUtilityEvaluator()
        self.generators: Dict[str, Any] = {}
        
    def train_generator(
        self, 
        data: pd.DataFrame, 
        generator_name: str,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Train a synthetic data generator."""
        logger.info(f"🚀 Training generator: {generator_name}")
        
        # Get configuration (mainly for logging and analysis)
        training_config = self.config.get_training_config(data)
        if custom_config:
            training_config.update(custom_config)
        
        # Prepare parameters for MOSTLY AI SDK
        # The SDK uses a simple API with mostly defaults
        train_params = {
            "name": generator_name,
            "data": data,
        }
        
        # Add any additional parameters that the SDK actually supports
        if custom_config:
            # Only include parameters the SDK actually accepts
            supported_params = ["name", "data"]  # Add more as SDK supports them
            for key in supported_params:
                if key in custom_config:
                    train_params[key] = custom_config[key]
        
        logger.info(f"🎯 Training with MOSTLY AI TabularARGN model...")
        logger.info(f"📊 Dataset: {data.shape[0]:,} rows × {data.shape[1]} columns")
        
        # Train the generator using the actual SDK API
        generator = self.mostly.train(**train_params)
        
        # Store generator
        self.generators[generator_name] = generator
        
        logger.info(f"✅ Generator {generator_name} trained successfully")
        return generator
    
    def generate_synthetic_data(
        self, 
        generator_name: str, 
        size: int,
        **generation_kwargs: Any
    ) -> pd.DataFrame:
        """Generate synthetic data using trained generator."""
        if generator_name not in self.generators:
            raise ValueError(f"Generator {generator_name} not found")
            
        generator = self.generators[generator_name]
        
        logger.info(f"📊 Generating {size:,} synthetic samples")
        
        # Use the correct SDK API for generation
        if size <= 10000:  # Use probe for smaller datasets (faster)
            synthetic_data = self.mostly.probe(generator, size=size, **generation_kwargs)
        else:  # Use generate for larger datasets
            synthetic_dataset = self.mostly.generate(generator, size=size, **generation_kwargs)
            synthetic_data = synthetic_dataset.data()
        
        logger.info(f"✅ Generated {len(synthetic_data):,} synthetic samples")
        return synthetic_data
    
    def evaluate_generator(
        self, 
        generator_name: str,
        original_data: pd.DataFrame,
        test_size: int = 1000
    ) -> Dict[str, float]:
        """Evaluate generator performance."""
        synthetic_data = self.generate_synthetic_data(generator_name, test_size)
        
        return self.evaluator.evaluate_privacy_utility(
            original_data, synthetic_data
        )
