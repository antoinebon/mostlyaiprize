"""Command-line interface for the competition."""

import click
import pandas as pd

from mostlyaiprize.core.config import CompetitionConfig
from mostlyaiprize.core.trainer import CompetitionTrainer


@click.group()
def cli() -> None:
    """MOSTLY AI Prize competition CLI."""
    pass


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--generator-name", default="competition_generator", help="Name for the generator")
@click.option("--privacy-level", default="medium", type=click.Choice(["low", "medium", "high"]))
def train_command(data_path: str, generator_name: str, privacy_level: str) -> None:
    """Train a synthetic data generator."""
    # Load data
    df = pd.read_csv(data_path)
    click.echo(f"📊 Loaded data: {df.shape}")
    
    # Initialize trainer
    config = CompetitionConfig(privacy_level=privacy_level)
    trainer = CompetitionTrainer(config)
    
    # Train generator
    generator = trainer.train_generator(df, generator_name)
    
    # Evaluate
    evaluation = trainer.evaluate_generator(generator_name, df)
    
    click.echo("🎯 Evaluation Results:")
    for metric, value in evaluation.items():
        click.echo(f"  {metric}: {value:.3f}")


@cli.command()
@click.argument("generator_name", default="competition_generator")
@click.option("--size", default=1000, help="Number of synthetic samples")
@click.option("--output", default="synthetic_data.csv", help="Output file path")
def generate_command(generator_name: str, size: int, output: str) -> None:
    """Generate synthetic data."""
    config = CompetitionConfig()
    trainer = CompetitionTrainer(config)
    
    # Generate data
    synthetic_data = trainer.generate_synthetic_data(generator_name, size)
    
    # Save
    synthetic_data.to_csv(output, index=False)
    click.echo(f"💾 Saved {len(synthetic_data)} samples to {output}")


@cli.command()
@click.argument("original_path", type=click.Path(exists=True))
@click.argument("synthetic_path", type=click.Path(exists=True))
def evaluate_command(original_path: str, synthetic_path: str) -> None:
    """Evaluate synthetic data quality."""
    from mostlyaiprize.evaluation.metrics import PrivacyUtilityEvaluator
    
    # Load data
    original_df = pd.read_csv(original_path)
    synthetic_df = pd.read_csv(synthetic_path)
    
    # Evaluate
    evaluator = PrivacyUtilityEvaluator()
    metrics = evaluator.evaluate_privacy_utility(original_df, synthetic_df)
    
    click.echo("📈 Evaluation Results:")
    for metric, value in metrics.items():
        click.echo(f"  {metric}: {value:.3f}")


if __name__ == "__main__":
    cli()
