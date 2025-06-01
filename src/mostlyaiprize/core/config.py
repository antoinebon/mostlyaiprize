"""Configuration management for the competition."""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CompetitionConfig:
    """Competition configuration settings."""
    
    # Mode settings
    local_mode: bool = True
    device: str = "cpu"
    random_seed: int = 42
    
    # Privacy settings
    privacy_level: str = "medium"  # low, medium, high
    enable_differential_privacy: bool = True
    
    # Performance settings
    time_budget_hours: float = 6.0
    max_epochs: int = 200
    early_stopping_patience: int = 15
    
    # Data settings
    train_test_split: float = 0.8
    min_dataset_size: int = 1000
    
    # Advanced settings
    ensemble_size: int = 1
    use_gpu_if_available: bool = True
    
    # Environment variables
    env_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Apply environment variable overrides."""
        # Load from environment
        self.privacy_level = os.getenv("PRIVACY_LEVEL", self.privacy_level)
        self.time_budget_hours = float(os.getenv("TIME_BUDGET_HOURS", self.time_budget_hours))
        self.random_seed = int(os.getenv("RANDOM_SEED", self.random_seed))
        self.device = os.getenv("DEVICE", self.device)
        
        # Apply any additional overrides
        for key, value in self.env_overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_training_config(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate training configuration based on data characteristics."""
        n_rows, n_cols = data.shape
        
        # Simple configuration matching MOSTLY AI SDK API
        config = {}
        
        # The SDK appears to use defaults and doesn't expose detailed configuration
        # We'll focus on what we can actually configure
        
        # Note: The MOSTLY AI SDK may not expose all these parameters
        # This is a placeholder for future SDK versions that might support more config
        self._log_recommended_settings(n_rows, n_cols)
        
        return config
    
    def _log_recommended_settings(self, n_rows: int, n_cols: int) -> None:
        """Log recommended settings for reference."""
        logger.info(f"📊 Dataset characteristics: {n_rows:,} rows × {n_cols} columns")
        
        # Log what we would configure if the API supported it
        recommended_epochs = self._get_adaptive_epochs(n_rows)
        recommended_batch_size = self._get_adaptive_batch_size(n_rows)
        recommended_lr = self._get_adaptive_learning_rate(n_rows)
        
        logger.info(f"🎯 Recommended settings (for reference):")
        logger.info(f"   Epochs: {recommended_epochs}")
        logger.info(f"   Batch size: {recommended_batch_size}")
        logger.info(f"   Learning rate: {recommended_lr}")
        logger.info(f"   Privacy level: {self.privacy_level}")
        logger.info("ℹ️  Using MOSTLY AI SDK defaults with TabularARGN model")
    
    def _get_adaptive_epochs(self, n_rows: int) -> int:
        """Get adaptive number of epochs based on dataset size."""
        if n_rows < 10000:
            return min(300, self.max_epochs)
        elif n_rows < 100000:
            return min(150, self.max_epochs)
        else:
            return min(80, self.max_epochs)
    
    def _get_adaptive_batch_size(self, n_rows: int) -> int:
        """Get adaptive batch size based on dataset size."""
        if n_rows < 10000:
            return min(64, max(8, n_rows // 100))
        elif n_rows < 100000:
            return min(128, max(16, n_rows // 500))
        else:
            return min(512, max(32, n_rows // 1000))
    
    def _get_adaptive_learning_rate(self, n_rows: int) -> float:
        """Get adaptive learning rate based on dataset size."""
        if n_rows < 10000:
            return 0.001
        elif n_rows < 100000:
            return 0.0005
        else:
            return 0.0003
    
    def _get_privacy_config(self, n_rows: int) -> Dict[str, Any]:
        """Get privacy configuration based on privacy level and dataset size."""
        privacy_configs = {
            "low": {
                "max_epsilon": 8.0,
                "noise_multiplier": 0.5,
                "max_grad_norm": 1.5,
                "delta": 1e-4
            },
            "medium": {
                "max_epsilon": 2.0,
                "noise_multiplier": 1.5,
                "max_grad_norm": 1.0,
                "delta": 1e-5
            },
            "high": {
                "max_epsilon": 0.5,
                "noise_multiplier": 3.0,
                "max_grad_norm": 0.5,
                "delta": 1e-6
            }
        }
        
        base_config = privacy_configs[self.privacy_level].copy()
        
        # Adjust delta based on dataset size
        base_config["delta"] = min(base_config["delta"], 1 / (n_rows ** 1.1))
        
        return base_config
