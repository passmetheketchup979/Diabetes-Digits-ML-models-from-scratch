# ECSE 551 — Assignment 1: ML Models from Scratch with NumPy

This project implements and evaluates four core machine learning algorithms — **Lasso Regression**, **ElasticNet Regression**, **Multiclass SVM (One-vs-Rest)**, and a **Multilayer Perceptron (MLP)** — entirely from scratch using NumPy, and benchmarks them against scikit-learn baselines.

## Overview

| Task | Dataset | Models | Baseline |
|---|---|---|---|
| Regression | Diabetes (442 patients, 10 features → 411 after outlier removal) | Lasso, ElasticNet (gradient descent + soft thresholding) | `sklearn.LinearRegression` |
| Classification | Digits (1,797 images, 64 pixels, 10 classes) | Multiclass SVM (OvR, hinge loss), MLP (ReLU + backprop) | `sklearn.RandomForestClassifier` |

## Key Results

- **Lasso / ElasticNet**: Test MSE ≈ 2840, matching sklearn's `LinearRegression` — confirming correct implementation.
- **Multiclass SVM**: Best test accuracy of **95.56%** at λ = 0.01, η = 0.001.
- **MLP**: Test accuracy of 0.87 (vs. RFC's 0.97 under comparable light tuning), converging to ~100% with sufficient epochs and hidden units.

## Methodology Highlights

### Lasso Regression
- Minimizes MSE + L1 penalty (`λ·Σ|w|`) for automatic feature selection.
- Uses **soft thresholding** to handle the non-differentiable L1 term at `w = 0`.

### ElasticNet Regression
- Combines L1 (sparsity/feature selection) and L2 (fast shrinkage) penalties.
- Reduces to Ridge when λ₁ = 0, and to Lasso when λ₂ = 0.

### Multiclass SVM
- One binary hinge-loss SVM per class (One-vs-Rest), with L2 (Ridge) regularization.
- Prediction via `argmax` over per-class decision scores.
- 70/30 train/test split, `random_state = 42`.

### MLP
- Single hidden layer, ReLU activation, softmax output, cross-entropy loss.
- Full forward and backward propagation derived and implemented manually.

## Hyperparameter Findings

- **Iterations**: MSE/accuracy improves then plateaus after convergence; more iterations beyond that point yield no benefit.
- **Learning rate (η)**: Too small → slow convergence; too large → oscillation/overshoot (e.g., SVM accuracy dropped from 95.56% to 90.56% at η = 0.01).
- **Regularization (λ, λ₁, λ₂)**: Classic bias-variance trade-off — too weak causes overfitting, too strong collapses the model (zeroed weights / underfitting).
- **Hidden layer size (MLP)**: More neurons generally improve accuracy up to a limit; overfitting was not clearly observed on this clean, relatively small dataset.

## Repository Contents

- `ECSE551_A1_Report.pdf` — full write-up with methodology, figures, tables, and discussion.

## Contributors

Tingyi Chen, Raisa Chowdhury, John Wu

## References

1. Harris et al., "Array programming with NumPy," *Nature*, 2020.
2. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, 2011.
3. Tibshirani, "Regression shrinkage and selection via the lasso," *JRSS-B*, 1996.
4. Zou & Hastie, "Regularization and variable selection via the elastic net," *JRSS-B*, 2005.
5. Cortes & Vapnik, "Support-vector networks," *Machine Learning*, 1995.
6. Rumelhart, Hinton & Williams, "Learning representations by back-propagating errors," *Nature*, 1986.
7. Tukey, *Exploratory Data Analysis*, 1977.
8. Murphy, *Probabilistic Machine Learning: An Introduction*, 2022.
