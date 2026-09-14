# Deep Learning Foundations: EMNIST Classification & Network Analysis

This repository contains the implementation of foundational Deep Learning models, moving from scratch implementations in NumPy to framework-based training in PyTorch[cite: 11, 12]. The project focuses on classifying handwritten letters from the EMNIST dataset and explores theoretical concepts surrounding activation functions[cite: 11, 12].

## Overview

The project is divided into three main components:
1. **Linear Classifiers & Custom MLP:** Building and training Perceptron, Logistic Regression, and Multi-Layer Perceptron models entirely from scratch using NumPy, including manual derivations of the backpropagation algorithm[cite: 11, 12].
2. **PyTorch FFN Analysis:** Utilizing PyTorch to build Feedforward Neural Networks (FFNs) to empirically study the Universal Approximation Theorem[cite: 11, 12]. This includes large-scale grid searches to analyze how varying hidden-layer widths and network depths affect interpolation and generalization[cite: 11, 12].
3. **Theoretical Analysis:** Mathematical proofs and derivations exploring sparse alternatives to the softmax function, specifically `sparsemax` and `relumax`[cite: 10, 12].

## Key Features

*   **Custom Feature Engineering:** Implements $2\times2$ average pooling to downsample $28\times28$ pixel images into 196 features, improving training efficiency by 2.8x with minimal accuracy loss[cite: 11].
*   **Hyperparameter Grid Search:** Automated scripts to tune learning rates, $l_{2}$ penalties, and dropout rates across multiple architectures[cite: 11].
*   **Model Checkpointing:** Custom logic to save model weights (`.pkl` files) whenever validation accuracy improves during training[cite: 11].
*   **Width vs. Depth Analysis:** Empirical evaluation showing that wider single-layer networks consistently improve training accuracy, while deeper networks (e.g., 9 layers) suffer from vanishing/exploding gradients[cite: 11].

## Repository Structure

The repository includes the following primary execution scripts:

**Question 1: From-Scratch Models**
*   `hw1-perceptron.py`: Trains a multi-class perceptron[cite: 11].
*   `hw1-logreg.py`: Trains a logistic regression classifier using Stochastic Gradient Descent (SGD) and $l_{2}$ regularization[cite: 11].
*   `hw1-logreg-b-downsample.py`: Trains the logistic regression model using the custom $2\times2$ downsampled feature representation[cite: 11].
*   `hw1-logreg-gridsearch.py`: Executes a grid search over learning rates, penalties, and feature representations[cite: 11].
*   `hw1-mlp.py`: Trains the custom Multi-Layer Perceptron (100 hidden units, ReLU activation) built from scratch[cite: 11].

**Question 2: PyTorch FFNs**
*   `hw1-ffn.py`: Base implementation of the PyTorch feedforward network[cite: 12].
*   `hw1_ffn_q2_a.py`: Grid search over 80 configurations for single-layer FFNs of varying widths (16 to 256 units)[cite: 11].
*   `hw1_ffn_q2_b.py`: Trains the overall best model (width 256) found in the grid search[cite: 11].
*   `hw1_ffn_q3_a.py` / `hw1_ffn_q3_b.py` / `hw1_ffn_q3_c.py`: Scripts to test the effect of depth by training models with 1, 3, 5, 7, and 9 layers[cite: 11].

## Dependencies

*   Python 3.x
*   NumPy (for from-scratch implementations)[cite: 12]
*   PyTorch (for FFN analysis)[cite: 12]
*   Matplotlib (for training curves)

## Dataset

The models are trained and evaluated on the **EMNIST Letter Dataset**, which consists of flattened 784-dimensional vectors representing $28\times28$ bitmap images of the 26 Roman alphabet letters[cite: 12]. 

## Usage Examples

**Train the Perceptron:**
```bash
python hw1-perceptron.py --save-path model.pkl
