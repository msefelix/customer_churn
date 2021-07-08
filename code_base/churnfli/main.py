from pakkr import Pipeline

from churnfli.prepare_data import load_data, clean_data, split_and_get_transformers, transform_train_test
from churnfli.feature_selection import feature_selection
from churnfli.train_test import load_train_test, grid_search_estimators, update_best_models
from churnfli.metrics import evaluate_models


def data_model_pipeline() -> Pipeline:
    """
    Return a data preparation and model training pipeline.
    """
    return Pipeline(load_data,
                    clean_data,
                    split_and_get_transformers,
                    transform_train_test,
                    feature_selection,
                    grid_search_estimators,
                    update_best_models,
                    _name="main")

def train_pipeline() -> Pipeline:
    """
    Return a pipeline for train and test models using preprocessed data.
    """
    return Pipeline(load_train_test,
                    grid_search_estimators,
                    update_best_models,
                    _name="train_test")

def main():
    data_model_pipeline()()

    GINIs, sensi_fig, churn_cat, churn_num_fig = evaluate_models()

    return GINIs, sensi_fig, churn_cat, churn_num_fig