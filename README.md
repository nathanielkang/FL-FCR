# FL-FCR: Federated Learning with Feature Calibration and Client Re-sampling

FL-FCR is an innovative Federated Learning (FL) framework designed to address data imbalance across clients in FL systems. This repository contains the implementation code for FL-FCR, including a demonstration and configuration files for running experiments on different datasets.

## Overview

In real-world FL scenarios, data across clients is often non-IID and imbalanced. FL-FCR tackles this challenge through a two-step process:
1. **Feature Calibration:** A calibrated loss function is integrated during client training to mitigate class imbalance.
2. **Client Re-sampling:** Global and local calibration statistics are used to guide data resampling on the server side, ensuring a more balanced representation of classes.

## Repository Structure

- `conf.py`: The configuration file where you can adjust parameters and specify which dataset to use for training.
- `FL-FCR Demo.ipynb`: A Jupyter notebook providing a hands-on demonstration of FL-FCR on a sample dataset.
- `src/`: The source code directory, including the implementation of FL-FCR.
- `data/`: Directory containing datasets used in the experiments.
- `models/`: Pre-defined models and training scripts.
- `utils/`: Utility functions for data processing, evaluation, and more.

## Quick Start

### Prerequisites

- Python 3.x
- Required Python packages are listed in `requirements.txt`. Install them using:
  ```bash
  pip install -r requirements.txt
