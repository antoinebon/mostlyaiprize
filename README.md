# MOSTLY AI Prize Submission

🏆 **Privacy-safe synthetic tabular data generation for the MOSTLY AI Prize**

## Overview

This repository contains our submission for the [MOSTLY AI Prize](https://www.mostlyaiprize.com/) - a $100,000 competition for creating the most accurate and privacy-safe synthetic tabular data.

## Quick Start

```bash
# Setup virtual env with dependencies
hatch shell

# Install the package in the virtual env
pip install -e .

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
* Configuration parameters (hydra config)
* Evaluation metrics (from qa report)
* System metrics (cpu usage, ...)
* Generated sequences
* MostlyAI QA reports (html)


## Development Setup

```bash
# Run tests
hatch run dev:test

# Format code
hatch run dev:format

# Run all quality checks
hatch run dev:lint
hatch run dev:typecheck
```

