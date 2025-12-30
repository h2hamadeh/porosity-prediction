# Deep Learning for Porosity Prediction
This repository contains a "from-scratch" implementation of a fully-connected feedforward neural network for predicting porosity from synthetic geophysical measurements in sandstone formations.

# Porosity Prediction Using Neural Networks

A from-scratch implementation of a fully-connected neural network for predicting reservoir porosity from acoustic impedance data. Built with pure NumPy to demonstrate deep understanding of backpropagation mechanics and neural network fundamentals.

## Overview

This project implements a multi-layer neural network **without using high-level ML frameworks** like TensorFlow or PyTorch.

The model predicts porosity (target variable) from:
- Acoustic Impedance (primary feature)
- Spatial coordinates (X, Y)

### Dataset

The project uses synthetic geophysical data, created by [**Professor Michael Pyrcz**](https://zenodo.org/records/5564874).

The dataset contains **480 data points** from sandstone formations with the following features (not all features were used in the model):
- **spatial (X,Y) coordinates** - predictive features
- **facies type** (Sandstone = 1, Shale = 0)
- **porosity** (0-1 range) - target variable
- **permeability**
- **acoustic impedance** - predictive feature

### Implementation Details

**Forward Propagation:**
```python
Z = W·A + b           # Linear transformation
A = max(0, Z)         # ReLU activation
```

**Backward Propagation:**
```python
dZ = dA * (Z > 0)     # ReLU gradient
dW = (1/m) * dZ·A^T   # Weight gradients
db = (1/m) * Σ(dZ)    # Bias gradients
```

**Parameter Update:**
```python
W = W - α·dW          # Update weights
b = b - α·db          # Update biases
```

## Key Functions

| Function | Purpose |
|----------|---------|
| `initialize_parameters()` | initialize weights with He initialization and biases to zero |
| `forward_linear()` | compute linear transformation Z = W·A + b |
| `relu()` | apply ReLU activation function |
| `forward_linear_activation()` | combined linear transformation + ReLU |
| `model_forward()` | full forward pass through all network layers |
| `compute_cost()` | calculate MSE loss between predictions and targets |
| `relu_backward()` | compute gradient through ReLU activation |
| `backward_linear()` | compute gradients for weights and biases |
| `backward_linear_activation()` | combined backward pass through activation + linear |
| `model_backward()` | full backward pass through all network layers |
| `update_parameters()` | apply gradient descent to update weights and biases |
| `layer_model()` | main training loop with validation monitoring |
| `model_test()` | evaluate trained model on test data |

## Project Structure
```
porosity-prediction/
├── data/
│   └── sample_data.csv
├── notebook/
│   ├── porosity_prediction.ipynb
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

## Tech Stack

- **Python 3.8+**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **SciPy**
- **Jupyter**

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/porosity-prediction.git
cd porosity-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your `sample_data.csv` file in the project directory

## Usage

### Running the Notebook

Open and run the Jupyter notebook:
```bash
jupyter notebook porosity_prediction.ipynb
```
## Future Work

- [ ] including more input features + feature slection and importance analysis
- [ ] training improvements and experimenting with architectures
- [ ] further hyperparameter tuning (e.g. Bayesian optimization)
- [ ] refactoring + congfiguration
