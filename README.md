# MOSTLY AI Prize Submission

🏆 **Privacy-safe synthetic tabular data generation for the MOSTLY AI Prize**

## Overview

This repository contains our submission for the [MOSTLY AI Prize](https://www.mostlyaiprize.com/) - a $100,000 competition for creating the most accurate and privacy-safe synthetic tabular data.

## Quick Start

```bash
# Install the package in a virtual env...
pip install -e .

# or use hatch to create one
hatch shell

# Train and generate sequences for sequential data challenge
mostlyai --config-name=config_flat

# Train and generate sequences for sequential data challenge
mostlyai --config-name=config_sequential
```

## Configuration

Runs are configured using hydra configuration system.
Configuration can be overriden by creating a new config file under `src/mostlyaiprize/conf` or by modifying the parameters in the existing config files, or by overriding the command line parameters.

## Experiment Tracking

Mlflow is used to track the following data:
* Configuration parameters
* Evaluation metrics
* System metrics
* Generated sequences
* MostlyAI QA reports


## Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
hatch run dev:setup-hooks

# Run all quality checks
hatch run dev:all
```

