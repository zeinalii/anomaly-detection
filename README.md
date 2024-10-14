# Credit Card Fraud Detection

## Introduction

Credit card fraud is a significant concern for financial institutions and customers alike. This project aims to detect fraudulent credit card transactions using machine learning techniques. It includes data preprocessing, model training, and deployment of a prediction API using FastAPI. Additionally, the project employs CI/CD practices with GitHub Actions to ensure robust testing and seamless deployment.

## Features

- **Data Preprocessing**: Handles imbalanced datasets, scales features, and removes outliers.
- **Model Training**: Supports Logistic Regression and Random Forest classifiers with hyperparameter tuning using GridSearchCV.
- **API Deployment**: Provides a FastAPI-based RESTful API for predicting fraud on new transactions.
- **CI/CD Pipeline**: Automates testing, model evaluation, and deployment using GitHub Actions.
- **Visualization**: Generates confusion matrices and other plots to visualize model performance.
- **Version Control with DVC**: Manages data and model versioning efficiently.

## Getting Started

Follow these instructions to set up and run the project on your local machine for development and testing purposes.

### Prerequisites

- **Python 3.8** or higher
- **Git**
- **Docker** (optional, for containerization)
- **AWS Account** (for S3 storage)
L