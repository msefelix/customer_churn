config = {
    "grid_parameters": {
        "Logistic": {
            "penalty": ["l2"],
            "C": [0.1, 1, 5, 10, 25, 50],
            "tol": [1e-7, 1e-5, 1e-4],
            "class_weight": [None, "balanced"],
            "max_iter": [100, 200, 500]
        },
        "XGB": {
            "max_depth": [2],
            "gamma": [30],
            "learning_rate": [0.3],
            "n_estimators": [100],
            "subsample": [0.5],
            "reg_alpha": [3],
            "reg_lambda": [3]
        },
        "RF": {
            "max_depth": [2, 3, 4],
            "n_estimators": [300, 1000, 3000],
            "max_features": [4, 6, 8],
            "min_samples_split": [10],
            "min_samples_leaf": [5],
            "class_weight": ["balanced"]
        },
    },
    "model_version": "v3",
    "test_size": 0.2,
    "model_types": ["Logistic", "RF", "XGB"],
    "scoring": "roc_auc",
}

# "XGB": {
# "max_depth": [2, 3, 4],
# "gamma": [30, 100],
# "learning_rate": [0.3, 1],
# "n_estimators": [100],
# "subsample": [0.5],
# "reg_alpha": [3, 10],
# "reg_lambda": [3, 10]
# },
