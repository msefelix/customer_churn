import pandas as pd

from pakkr import returns
from typing import Dict, List
from sklearn import model_selection
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from IPython.display import display

from churnfli.utils import save_model, load_model
from churnfli.settings import config
from churnfli.metrics import compute_gini


def init_estimator(name: str) -> BaseEstimator:
    if name == 'SVM':
        estimator = SVC()
    elif name == 'Logistic':
        estimator = LogisticRegression(solver="lbfgs", fit_intercept=True)
    elif name == "XGB":
        estimator = XGBClassifier()
    elif name == "RF":
        estimator = RandomForestClassifier()
    return estimator


def grid_search_estimator(features: pd.DataFrame, labels: pd.Series,
                          estimator: BaseEstimator, parameters: List,
                          model_name: str, scoring: str) -> BaseEstimator:

    print("Performing grid search...")
    estimator = model_selection.GridSearchCV(estimator=estimator,
                                             scoring=scoring,
                                             param_grid=parameters,
                                             cv=5,
                                             n_jobs=-1,
                                             verbose=1)
    estimator.fit(features, labels)
    save_model(estimator, f"./models/{model_name}")
    return estimator


@returns(train_features=pd.DataFrame,
         train_labels=pd.Series,
         test_features=pd.DataFrame,
         test_labels=pd.Series)
def load_train_test():
    df_train = pd.read_parquet("./intermediate/df_train.parquet")
    df_test = pd.read_parquet("./intermediate/df_test.parquet")

    return {
        "train_features": df_train.drop('churn', axis=1),
        "train_labels": df_train['churn'],
        "test_features": df_test.drop('churn', axis=1),
        "test_labels": df_test['churn']
    }


@returns()
def grid_search_estimators(train_features: pd.DataFrame,
                           train_labels: pd.Series) -> None:
    """
    Grid Serch to find out the best hyperparameters for different algorithms and save the gridsearch object.
    """

    estimators = {}
    for model_type in config["model_types"]:
        print("##############################################################")
        print(f"###################### Start to train {model_type} models... ")
        print("##############################################################")
        estimator = init_estimator(model_type)
        parameters = config["grid_parameters"][model_type]
        model_name = f'{config["model_version"]}_{model_type}'

        estimator = grid_search_estimator(train_features,
                                          train_labels,
                                          estimator,
                                          parameters,
                                          model_name,
                                          scoring=config["scoring"])

        estimators[model_type] = estimator

    return


def update_best_models(train_features: pd.DataFrame, train_labels: pd.Series,
                       test_features: pd.DataFrame,
                       test_labels: pd.Series) -> Dict:
    """
    Re-train, on the full training dateset, the best model in each type by using the optimised hyperparamters from grid search.
    Save the re-trained models and check their performance on the test data.
    """
    scores = pd.DataFrame(columns=['train', 'test'])

    for model_type in config["model_types"]:
        # Load best hyperparameter
        estimator = init_estimator(model_type)
        model_name = f"{config['model_version']}_{model_type}"
        estimator.set_params(**load_model(f"models/{model_name}").best_params_)

        # Re-train on full training data (80%), instead of the 80% * 80% used in grid search under CV=5.
        estimator.fit(train_features, train_labels)

        # Save updated model
        save_model(estimator, f"./models/final/{model_name}_final")

        # Get scores of the re-trained models
        scores.loc[f"{model_name}_final", 'train'] = compute_gini(
            train_labels,
            estimator.predict_proba(train_features)[:, 1])
        scores.loc[f"{model_name}_final", 'test'] = compute_gini(
            test_labels,
            estimator.predict_proba(test_features)[:, 1])

    # Print scores to select the best model for serving
    scores = scores.sort_values('test', ascending=False)
    print("#####################################################")
    print(
        f"GINI on the train and test data for the re-trained best models are:")
    print("#####################################################")
    display(scores)
    scores.to_csv(f"./models/{config['model_version']}_GINI.csv")

    return
