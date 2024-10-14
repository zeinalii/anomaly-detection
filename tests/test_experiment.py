import os


def check_model_file():
    assert os.path.exists("assets/trained_model.pickle")


def check_metrics_file():
    assert os.path.exists("assets/model_metrics.json")


def check_dataset_file():
    assert os.path.exists("assets/credit_data.csv")
