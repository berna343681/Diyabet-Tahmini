import pandas as pd
import numpy as np


def load_data(file_path):
    return pd.read_csv(file_path)


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def handle_missing_values(dataframe, zero_columns):
    for col in zero_columns:
        dataframe[col] = np.where(dataframe[col] == 0, np.nan, dataframe[col])
    return dataframe


def fill_missing_values_with_median(dataframe, zero_columns):
    for col in zero_columns:
        dataframe.loc[dataframe[col].isnull(), col] = dataframe[col].median()
    return dataframe
