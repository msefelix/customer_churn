import pandas as pd
import plotly.express as px

from sklearn.metrics import roc_auc_score
from IPython.display import display

from churnfli.settings import config
from churnfli.utils import load_pickle, load_model

# Load feature list (with sequence) used to train the models
dev_cols = load_pickle("./intermediate/dev_cols.pickle")


def pred_churn_prob(estimator, df):
    """Predict churn probability.

    Args:
        estimator: Trained estimator with API similar to sklearn.base.BaseEstimator
        df (pd.DataFrame): Table containing transformed features.

    Returns:
        pd.Series: Indexed by the input table's index and the value is the probablity of churn
    """
    return pd.Series(index=df.index,
                     data=estimator.predict_proba(df[dev_cols])[:, 1])


def compute_gini(y_true, y_predict):
    """Compute GINI using label and predicted probability with array-like inputs.
    """    
    return roc_auc_score(y_true, y_predict) * 2 - 1


def sensitivity_curve(df_in,
                      y_col,
                      pred_col,
                      predict_round=0,
                      reverse_outcome=False):
    """Returns sensitivity curve data for the given labels and predictions.

    Args:
        df_in (pd.DataFrame): Table containing label and prediction columns
        y_col (str): name of the label column
        pred_col (str): name of the prediction column
        predict_round (int, optional): Rounding digits for the prediction. Defaults to 0, which means no rounding.
        reverse_outcome (bool, optional): Whether to reverse 1/0 label to match the label transformation used in development. Defaults to False.

    Returns:
        [type]: [description]
    """
    df = df_in[[y_col, pred_col]].copy().dropna(how='any')

    if reverse_outcome:
        df[y_col] = 1 - df[y_col]

    if predict_round > 0:
        df[pred_col] = df[pred_col].round(predict_round)

    df["record"] = 1
    df["bad"] = 1 - df[y_col]

    df = df.groupby(pred_col).sum().sort_index().drop(y_col, axis=1)

    df['cumulative bad %'] = (df['bad'].cumsum() /
                              df['bad'].sum()).fillna(0) * 100
    df['cumulative total %'] = (df['record'].cumsum() /
                                df['record'].sum()) * 100

    df['pred_col'] = pred_col

    return df
    
def evaluate_models():
    df = consolidate_pred()
    GINIs, sensi_fig = evaluate_pred(df)
    churn_cat, churn_num_fig = churn_rate_summary(df)
    return GINIs, sensi_fig, churn_cat, churn_num_fig

def consolidate_pred():
    """Get Predictions for train&test data using all trained models (as defined in config).
       Also load the original features.
    """    
    df_train = pd.read_parquet("./intermediate/df_train.parquet")
    df_train['train_test'] = 'train'
    df_test = pd.read_parquet("./intermediate/df_test.parquet")
    df_test['train_test'] = 'test'
    df = pd.concat([df_train, df_test], axis=0, ignore_index=False)

    for model_type in config['model_types']:
        estimator = load_model(
            f"models/final/{config['model_version']}_{model_type}_final")
        df[f'prob_{model_type}'] = pred_churn_prob(estimator, df)

    df_ori = pd.read_parquet("./intermediate/df_cleaned.parquet").drop('churn',axis=1)
    df = df.drop(dev_cols, axis=1).join(df_ori)

    return df


def evaluate_pred(df):
    """For all models:
    a. Compute GINI for train data and test data, respectively.
    b. Compute sensitivity curve for all data.
    """
    pred_cols = [f"prob_{x}" for x in config['model_types']]
    groups = df[['churn', 'train_test'] + pred_cols].groupby('train_test')

    # GINI results
    GINIs = groups.apply(lambda dfx: pd.Series(
        data=[compute_gini(dfx['churn'], dfx[col]) for col in pred_cols],
        index=pred_cols))

    # Sensitivity results
    sensi = [
        sensitivity_curve(df, 'churn', col, predict_round=3)
        for col in pred_cols
    ]
    sensi = pd.concat(sensi, axis=0)
    sensi.to_csv("./intermediate/sensitivity.csv")

    fig = px.line(sensi,
                  y='cumulative bad %',
                  x='cumulative total %',
                  color='pred_col',
                  width=800,
                  height=800,
                  title="Sensitivity of final models' predictions")
    fig.write_html("./intermediate/sensitivity.html")

    return GINIs, fig


def churn_rate_summary(df):
    """Prepare churn rates per group per feature.
    For numercial features, deciles are used to replace the raw values

    Returns:
        tuple: A table for categorical features and a fig for numerical features.
    """
    ### Prepare churn rates of categorical variables
    # Load feature importance from the best RF model
    rf_best_model = load_model(f"models/final/v3_RF_final")
    fi = pd.Series(data=rf_best_model.feature_importances_, index=dev_cols)
    fi = fi.sort_values(ascending=False).to_frame('feature importance')

    # Load target codes of categorical variables from training data
    target_codes = load_pickle("./intermediate/target_codes.pickle")
    codes_df = []
    for col, codes in target_codes.items():
        tt = pd.Series(codes).reset_index().rename(columns={
            'index': 'value',
            0: 'churn rate (%)'
        })
        tt['churn rate (%)'] = ((1 - tt['churn rate (%)']) * 100).round(1)
        tt['feature'] = col
        codes_df.append(tt)
    codes_df = pd.concat(codes_df)

    # Combine the two
    rates_cat = fi.join(codes_df.set_index('feature'), how='inner')
    rates_cat = rates_cat.sort_values(["feature importance", "churn rate (%)"],
                                      ascending=False)
    print("\n Feature importance and churn rates for categorical features:")
    display(rates_cat)

    ### Prepare churn rates of numerical variables
    num_cols = ['monthlycharges', 'totalcharges', 't_m_ratio', 'tenure']

    # Convert raw values to deciles
    rates_num = pd.DataFrame(
        {col: pd.qcut(df[col], 10, labels=False) for col in num_cols})
    rates_num = rates_num.join(df.loc[df['train_test'] == 'train', 'churn'],
                               how='inner')

    # Calculate churn rate by decile
    rates_num = {
        col: (1 - rates_num[[col, 'churn']].groupby(col).mean()['churn']) * 100
        for col in num_cols
    }
    rates_num = pd.DataFrame(rates_num)
    rates_num.index.name = "decile"
    rates_num.index = rates_num.index + 1

    # Make a plot
    fig = px.line(rates_num, title="Churn rate by decile for numerical features")
    fig.to_html("./intermediate/churn_rate_num_features.html")

    # Print feature importance
    print("\n Feature importance for numerical features:")
    display(fi.loc[num_cols])

    return rates_cat, fig
