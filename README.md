# Car Price Prediction - Assignment 2

## Overview
This project extends Assignment 1's car price prediction with a custom Linear Regression implementation built from scratch, including regularization (Ridge, Lasso), Xavier initialization, momentum, and feature importance visualization.

## Results
- 144 experiments conducted using MLflow (4 model types × 2 momentum × 3 GD methods × 2 init methods × 3 learning rates)
- Best model: Normal (no regularization), stochastic GD, zeros init, lr=0.001, no momentum
- Test R2: 0.884 | Test MSE: 0.065

## Project Structure
- `notebooks/car-prediction.ipynb` — Main notebook with preprocessing, custom class, experiments, and report
- `app/` — Dash web application with two models (old Random Forest + new Linear Regression)
- `data/` — Dataset and experiment results
- `docs/` — MLflow screenshots

## Deployment
The web app is deployed at: https://web-st127401.ml.brain.cs.ait.ac.th

## How to Run Locally
```bash
cd app
docker compose up --build
```
Then visit http://localhost:8050
