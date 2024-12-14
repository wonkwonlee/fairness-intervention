# Fairness-Enhancing Interventions in Machine Learning Pipelines

This repository demonstrates the application of fairness-enhancing techniques in machine learning pipelines using the `aif360` library and `RandomForestClassifier`. It evaluates models for fairness using metrics such as False Positive Rate Difference and Disparate Impact.

## Features

- **Fairness Metrics Calculation**: Evaluate models on metrics like False Positive Rate and Disparate Impact.
- **Baseline Random Forest Model**: Train and assess a simple Random Forest pipeline.
- **Extensible Framework**: Integrate fairness-enhancing techniques such as pre-processing, in-processing, and post-processing.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/wonkwonlee/fairness-intervention.git
cd fairness-pipeline
pip install -r requirements.txt
```

## Usage
python fairness_pipeline.py

## Key Metrics
The framework evaluates machine learning models using the following fairness metrics:

1. False Positive Rate (Privileged/Unprivileged)
2. False Positive Rate Difference
3. Overall Accuracy
4. Group-specific Accuracy


