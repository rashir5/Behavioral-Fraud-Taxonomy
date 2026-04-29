# Behavioral Fraud Detection
Multi-class fraud detection pipeline using behavioral features, velocity profiling, and SHAP-based explainability.

## Overview
This project implements a machine learning system to detect 11 different types of credit card fraud. Unlike traditional binary classification, it focuses on identifying behavioral patterns such as sudden spending spikes and unusual transaction locations.

## Tech Stack
- Python  
- XGBoost, LightGBM  
- Pandas, NumPy  
- SHAP (Explainability)

## Key Features
- Multi-class fraud detection (11 categories)  
- Behavioral feature engineering (transaction velocity, location anomalies)  
- High-performance models achieving 0.998 AUC  
- Model interpretability using SHAP values  

## Project Structure
- fraud_detection_research.ipynb – data preprocessing, feature engineering, model training, and evaluation  
- requirements.txt – project dependencies  
- data/ – directory for dataset (not included due to size constraints)

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Download the dataset:
   - Obtain the dataset (e.g., fraudTest.csv) from Kaggle  
   - Place it inside the `data/` folder  

3. Run the project:
   - Open `fraud_detection_research.ipynb` in Google Colab or Jupyter Notebook  
   - Execute all cells sequentially  

## Explainability
SHAP values are used to interpret model predictions and identify which features (such as transaction amount or distance from home) contribute most to fraud classification.

## Future Work
- Deploy as a real-time fraud detection system  
- Improve generalization on unseen datasets  
- Add visualization dashboard  
