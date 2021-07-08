import pandas as pd
import plotly.express as px
from typing import Dict, List
from pakkr import returns, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from churnfli.utils import save_pickle
from churnfli.settings import config


@returns(pd.DataFrame)
def load_data():
    df = pd.read_csv("./data/churn_challenge.csv", index_col='customerID')
    return df


@returns(pd.DataFrame)
def clean_data(df):
    # Remove a tiny proportion of records with tenure and TotalCharges equals to 0
    # Note that their labels are all 'no churn' thus the impact of simply dropping them should be minor
    df = df[df['tenure'] != 0]
    df['TotalCharges'] = df['TotalCharges'].astype(float)

    # Convert the column 'SeniorCitizen' to object as it only has two unique values
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

    # Convert column names to lower string for consistency
    df.columns = [x.lower() for x in df.columns]

    # Convert target to numbers
    df['churn'] = df['churn'].map({'No': 1, 'Yes': 0})

    # Create a feature: ratio of 'totalcharges' over 'monthlycharges'
    # It is similar to 'tenure' but includes some information on the actual payment
    df['t_m_ratio'] = df['totalcharges'] / df['monthlycharges']

    # Save for later use
    df.to_parquet("./intermediate/df_cleaned.parquet")

    return df


@returns(pd.DataFrame, pd.DataFrame, Dict, List, StandardScaler)
def split_and_get_transformers(df):
    # Get lists of object and numerical columns
    obj_cols = df.dtypes[df.dtypes == 'O'].index.tolist()
    num_cols = df.dtypes[df.dtypes != 'O'].index.tolist()
    num_cols.remove('churn')

    # Split train and test
    df_train, df_test = train_test_split(df,
                                         test_size=config['test_size'],
                                         random_state=0,
                                         stratify=df['churn'])

    # Calculate target codes for obj_cols on training data
    target_codes = {
        col: df_train[[col, 'churn']].groupby(col).mean()['churn'].to_dict()
        for col in obj_cols
    }

    # Get normaliser for num_cols on training data
    scaler = StandardScaler()
    scaler.fit(df_train[num_cols])

    # Save artifacts for future use (predict other samples in future)
    save_pickle(target_codes, "./intermediate/target_codes.pickle")
    save_pickle(num_cols, "./intermediate/num_cols.pickle")
    save_pickle(scaler, "./intermediate/num_scaler.pickle")

    return df_train, df_test, target_codes, num_cols, scaler


def _transform_a_df(df, target_codes, num_cols, scaler):

    for col, reps in target_codes.items():
        df.loc[:, col] = df[col].replace(to_replace=reps)

    df.loc[:, num_cols] = scaler.transform(df[num_cols])

    return df


@returns(pd.DataFrame, pd.DataFrame)
def transform_train_test(df_train, df_test, target_codes, num_cols, scaler):
    df_train = _transform_a_df(df_train, target_codes, num_cols, scaler)
    df_test = _transform_a_df(df_test, target_codes, num_cols, scaler)

    # Ensure no null values after transformation
    null_train = pd.Series(
        index=df_train.columns,
        data=[df_train[col].isna().sum() for col in df_train.columns])
    null_test = pd.Series(
        index=df_test.columns,
        data=[df_test[col].isna().sum() for col in df_test.columns])
    null_df = pd.concat([null_train, null_test], axis=1)
    null_df.columns = ['train', 'test']
    if len(null_df[null_df.sum(axis=1) != 0]) > 0:
        print("Null values found in some features:")
        print(null_df[null_df.sum(axis=1) != 0])

    # Check distribution of features to ensure they are on the similar orders of magnitude
    fig = px.line(df_train.describe().T.drop('count', axis=1))
    fig.write_html("./intermediate/feature_dist_after_transformation.html")

    # Save for later use
    df_train.to_parquet("./intermediate/df_train_bf_fs.parquet")
    df_test.to_parquet("./intermediate/df_train_bf_fs.parquet")

    return df_train, df_test


def prepare_data_pipeline() -> Pipeline:
    """
    Return a PAKKR pipeline for preprocessing data.
    """
    return Pipeline(load_data,
                    clean_data,
                    split_and_get_transformers,
                    transform_train_test,
                    _name="prepare_data")
