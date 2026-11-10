# Project 2: Neural Networks and Deep Learning

This project focuses on building and understanding neural networks from scratch. The central goal is to implement a feed-forward neural network with backpropagation and use it to study both regression and classification problems. The work is designed to connect directly with the earlier project, but now the trainable parameters are the network weights and biases instead of the polynomial coefficients.

The main idea is to analyze model performance using mean squared error (MSE) and classification losses, with and without regularization. In this project, the penalty terms are applied to the weight matrices rather than to a coefficient vector such as theta.

## How to run the project

## Project structure

### Main notebook

- `Project2.ipynb`: the main notebook for the project. It contains the overall workflow, experiments, and reporting for the neural-network assignment.

### Supporting scripts

- `requirements.txt`: required Python packages for the project.
- `submit-parts/data_aquisition.py`: data loading and synthetic data generation utilities.
- `submit-parts/function_setup2.py`: activation functions, losses, and reusable helper functions such as Runge data generation and basic numerical utilities.
- `submit-parts/part_a.py`: cost functions, penalties, and derivative definitions used in the neural-network optimization process.
- `submit-parts/part_b.py`: forward pass, backpropagation, model training, and optimizer implementations.
- `submit-parts/part_c.py`: comparison of network architectures and activation functions, including training and test error evaluation.
- `submit-parts/part_d.py`: regularized training and hyperparameter tuning for neural networks.
- `submit-parts/part_e.py`: multi-class classification using MNIST and metrics such as confusion matrices and ROC analysis.
- `submit-parts/verifications.py`: validation and sanity-check routines for the project implementation.

### Generated results

- `plots/`: saved figures and visual outputs from the experiments, including learning-rate sweeps, MSE comparison plots, ROC plots, and confusion matrices.

## What each part is for

### Part A
This part defines the optimization objective. It includes the cost functions, MSE definitions, penalty terms, and the derivatives needed for training.

### Part B
This part builds the training machinery: forward propagation, backpropagation, network initialization, and optimizers such as SGD, RMSprop, and Adam.

### Part C
This part investigates how architecture and activation functions affect performance. It compares shallow and deeper models and tests different choices such as sigmoid, ReLU, and leaky ReLU.

### Part D
This part introduces regularized training. It searches over regularization strengths and learning rates to improve generalization and compare model behavior against the baseline.

### Part E
This part focuses on a real classification task using MNIST digits. It trains a softmax output network, evaluates performance, and produces classification metrics.

## How to use this project

Use this folder as a numerical experimentation project rather than as a web app or package. In practice:

1. Open the notebook to understand the full workflow, or run the part scripts in sequence.
2. Start with the simpler modules (`part_a.py` and `part_b.py`) to understand the losses and training logic.
3. Move on to `part_c.py` and `part_d.py` to compare architectures and regularization settings.
4. Use `part_e.py` for the MNIST classification task and metric analysis.
5. Inspect the figures in `plots/` to understand the results visually.

The code is intended to be educational: it is built to be read, run, and modified so you can test different architectures, activations, learning rates, and regularization strategies.

## LLM declaration

This project includes code and generated visual outputs that were developed with the assistance of large language models for implementation support, explanation, and iterative debugging. In particular, a substantial amount of the plotting and reporting work for the project, especially the results associated with part E and the generated figures in `plots/`, was produced to support understanding and validation of the experiments.

## Summary

This folder is the implementation of a deep-learning assignment focused on neural-network theory and practice. The repository is designed for experimentation and learning, not deployment, and the best way to use it is to run the scripts, inspect the plots, and adjust the code to test your own hypotheses.

