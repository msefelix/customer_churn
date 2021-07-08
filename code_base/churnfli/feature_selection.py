import pandas as pd
import plotly.express as px

from pprint import pprint
from pakkr import returns
from sklearn.ensemble import RandomForestClassifier

from churnfli.utils import save_pickle


@returns(train_features=pd.DataFrame,
         train_labels=pd.Series,
         test_features=pd.DataFrame,
         test_labels=pd.Series)
def feature_selection(df_train, df_test):
    print("#####################################################")
    print("Perform rough feature selection with a RF model...")
    print("#####################################################")

    # Check correlation
    fig = px.imshow(df_train.corr().abs(), color_continuous_scale='Viridis')
    fig.write_html("./intermediate/feature_correlation_on_train.html")

    # Train a quick RF estimator to check feature importance
    estimator = RandomForestClassifier(n_estimators=10000,
                                       max_depth=4,
                                       min_samples_split=20,
                                       min_samples_leaf=10,
                                       random_state=1,
                                       class_weight='balanced',
                                       n_jobs=3)
    estimator.fit(df_train.drop('churn', axis=1), df_train['churn'])
    print("Mean accuracy on the train data is ",
          estimator.score(df_train.drop('churn', axis=1), df_train['churn']))

    # Check feature importance
    fi = pd.Series(data=estimator.feature_importances_,
                   index=df_train.drop('churn', axis=1).columns)
    fi = fi.sort_values(ascending=False).to_frame('feature importance')
    fi['cumulative feature importance'] = fi['feature importance'].cumsum()
    fig = px.line(fi)
    fig.write_html("./intermediate/feature_importance.html")

    # Drop trival features
    trivial_features = fi[
        fi['cumulative feature importance'] >= 0.99].index.tolist()
    print(f"These trival features will be excluded from modelling:")
    pprint(trivial_features)

    # Save for future use
    save_pickle(trivial_features, "./intermediate/trivial_features.pickle")

    # Remove trival features
    df_train = df_train.drop(trivial_features, axis=1)
    df_test = df_test.drop(trivial_features, axis=1)

    # Save for later use
    df_train.to_parquet("./intermediate/df_train.parquet")
    df_test.to_parquet("./intermediate/df_test.parquet")
    save_pickle(
        df_train.drop('churn', axis=1).columns.tolist(),
        "./intermediate/dev_cols.pickle")

    return {
        "train_features": df_train.drop('churn', axis=1),
        "train_labels": df_train['churn'],
        "test_features": df_test.drop('churn', axis=1),
        "test_labels": df_test['churn']
    }
