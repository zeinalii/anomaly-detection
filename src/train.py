# Reference: Kaggle Dataset on Credit Fraud - https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets

import json
import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from preprocessing import prepare_dataset


def run_training_pipeline():
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    dataset_path = os.path.join(data_dir, "creditcard.csv")
    dataset = pd.read_csv(dataset_path)
    processed_data = prepare_dataset(dataset, training=False)
    
    X_train = processed_data["X_train"]
    X_validation = processed_data["X_valid"]
    y_train = processed_data["y_train"]
    y_validation = processed_data["y_valid"]
    
    selected_model = "logistic_regression"
    
    if selected_model == "logistic_regression":
        model = LogisticRegression(solver='liblinear')  # Specifying solver for compatibility
        param_grid = {
            "penalty": ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
    else:
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [500],
            'max_depth': [50],
            'max_leaf_nodes': [100]
        }
    
    print(f"Initializing model: {model}")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=2,
        refit=True
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    predictions = best_model.predict(X_validation)
    report = classification_report(y_validation, predictions, output_dict=True)
    
    conf_matrix = confusion_matrix(y_validation, predictions, normalize='true')
    heatmap = sns.heatmap(conf_matrix, annot=True, fmt='.2%', cmap='Blues')
    heatmap_fig = heatmap.get_figure()
    heatmap_fig.savefig(os.path.join(data_dir, 'confusion_matrix.png'), dpi=400)
    
    evaluation_metrics = {
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
    }
    
    metrics_path = os.path.join(data_dir, 'metrics.json')
    with open(metrics_path, 'w') as metrics_file:
        json.dump(evaluation_metrics, metrics_file, indent=4)
    
    model_path = os.path.join(data_dir, 'model.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(best_model, model_file)
    
    print("Training pipeline completed successfully.")


if __name__ == '__main__':
    run_training_pipeline()
