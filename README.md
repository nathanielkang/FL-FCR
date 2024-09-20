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


### Example Configuration in `conf.py`

Adjust the configuration settings in `conf.py` to specify the dataset and other training parameters. Here is an example configuration:

```python
# conf.py

# Dataset selection
dataset = 'cifar10'  # Choose from 'mnist', 'fmnist', 'cifar10', 'cifar100'

# Federated Learning parameters
num_clients = 20
local_epochs = 5
communication_rounds = 50
learning_rate = 0.01
alpha = 0.05  # Dirichlet distribution parameter for data partitioning

# Model and training configurations
batch_size = 32
optimizer = 'SGD'
momentum = 0.9

# Device configuration
use_gpu = True
gpu_device = 'cuda:0'  # Set to 'cpu' if GPU is not available

```


## Citation

The skeleton of this code is derived from the following paper:

```bibtex
@inproceedings{10.5555/3540261.3540718,
author = {Luo, Mi and Chen, Fei and Hu, Dapeng and Zhang, Yifan and Liang, Jian and Feng, Jiashi},
title = {No fear of heterogeneity: classifier calibration for federated learning with non-IID data},
year = {2024},
isbn = {9781713845393},
publisher = {Curran Associates Inc.},
address = {Red Hook, NY, USA}, 
booktitle = {Proceedings of the 35th International Conference on Neural Information Processing Systems},
articleno = {457},
numpages = {13},
series = {NIPS '21}
}

We have made substantial modifications to extend and improve upon the original framework.
