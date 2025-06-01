# MOSTLY AI Prize Submission

🏆 **Privacy-safe synthetic tabular data generation for the MOSTLY AI Prize**

## Overview

This repository contains our submission for the [MOSTLY AI Prize](https://www.mostlyaiprize.com/) - a $100,000 competition for creating the most accurate and privacy-safe synthetic tabular data.

## Quick Start

```bash
# Install the package
pip install -e .

# Download competition data
hatch run dev:download-data

# Train on Challenge 1
hatch run dev:train-challenge1

# Generate submission
hatch run dev:generate-submission
```

## Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
hatch run dev:setup-hooks

# Run all quality checks
hatch run dev:all
```

## Project Structure

```
mostlyaiprize/
├── src/mostlyaiprize/          # Main package
│   ├── core/                   # Core training and generation logic
│   ├── models/                 # Model architectures and configurations
│   ├── evaluation/             # Evaluation metrics and privacy analysis
│   ├── utils/                  # Utility functions and helpers
│   └── cli/                    # Command-line interface
├── tests/                      # Test suite
├── scripts/                    # Competition scripts
├── notebooks/                  # Jupyter notebooks for exploration
├── data/                       # Competition datasets (gitignored)
└── submissions/                # Generated submissions
```

## Competition Strategy

Our approach focuses on:

1. **TabularARGN Architecture**: Leveraging MOSTLY AI's state-of-the-art autoregressive model
2. **Adaptive Privacy Configuration**: Dynamic epsilon/delta tuning based on dataset characteristics
3. **Multi-objective Optimization**: Balancing privacy-utility tradeoff through systematic evaluation
4. **Ensemble Methods**: Combining multiple models for improved robustness

## Evaluation Metrics

- **Utility**: Total Variation Distance (TVD) across marginal distributions
- **Privacy**: Distance-based privacy analysis and differential privacy guarantees
- **Performance**: Generation speed and computational efficiency

## License

MIT License - see [LICENSE](LICENSE) for details.

## Competition Timeline

- **Start**: May 14, 2025
- **End**: July 3, 2025
- **Prize**: $100,000 ($50k per challenge)

---

*Building the future of privacy-safe data sharing* 🔒📊
