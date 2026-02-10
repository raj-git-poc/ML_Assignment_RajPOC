# Heart Disease Prediction using Machine Learning

## Problem Statement
The objective of this project is to build and deploy multiple machine learning classification models to predict the presence of heart disease using clinical patient data.

## Dataset Description
The dataset used is the Heart Disease UCI dataset obtained from Kaggle. It consists of 1025 patient records with 13 clinical features such as age, cholesterol level, blood pressure, and maximum heart rate achieved. The target variable indicates whether the patient has heart disease (1) or not (0).

## Models Used and Evaluation Metrics

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|---------|-----|----------|-------|----|----|
| Logistic Regression |  |  |  |  |  |  |
| Decision Tree |  |  |  |  |  |  |
| KNN |  |  |  |  |  |  |
| Naive Bayes |  |  |  |  |  |  |
| Random Forest |  |  |  |  |  |  |
| XGBoost |  |  |  |  |  |  |

(Fill values from model/model_metrics.csv)

## Model Performance Observations

| Model | Observation |
|------|------------|
| Logistic Regression | Provides stable baseline performance |
| Decision Tree | Prone to overfitting |
| KNN | Sensitive to feature scaling |
| Naive Bayes | Fast but assumes feature independence |
| Random Forest | Strong ensemble performance |
| XGBoost | Best overall accuracy and robustness |

## Deployment
The application is deployed using Streamlit Community Cloud and provides an interactive interface for dataset upload, model selection, performance metrics, and confusion matrix visualization.

