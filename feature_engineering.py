import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def create_new_features(dataframe):
    dataframe["NEW_AGE_CAT"] = pd.cut(dataframe["Age"], bins=[21, 30, 40, 50, 60, 70, 80],
                                      labels=["21-30", "30-40", "40-50", "50-60", "60-70", "70-80"])
    dataframe.loc[(dataframe['BMI'] < 18.5), 'NEW_BMI_CAT'] = 'underweight'
    dataframe.loc[(dataframe['BMI'] >= 18.5) & (dataframe['BMI'] <= 24.9), 'NEW_BMI_CAT'] = 'healthy'
    dataframe.loc[(dataframe['BMI'] >= 25) & (dataframe['BMI'] <= 29.9), 'NEW_BMI_CAT'] = 'overweight'
    dataframe.loc[(dataframe['BMI'] >= 30), 'NEW_BMI_CAT'] = 'obese'
    dataframe.loc[(dataframe['Glucose'] >= 140), 'NEW_GLUCOSE'] = 'High'
    dataframe.loc[(dataframe['Glucose'] >= 70) & (dataframe['Glucose'] < 140), 'NEW_GLUCOSE'] = 'Normal'
    dataframe.loc[(dataframe['Glucose'] < 70), 'NEW_GLUCOSE'] = 'Low'
    dataframe['NEW_INSULIN_SCORE'] = dataframe.apply(lambda x: 'Normal' if 16 <= x['Insulin'] <= 166 else 'Abnormal',
                                                     axis=1)
    return dataframe


def encode_features(dataframe, binary_cols, cat_cols):
    for col in binary_cols:
        labelencoder = LabelEncoder()
        dataframe[col] = labelencoder.fit_transform(dataframe[col])
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=True)
    return dataframe


def scale_features(dataframe, num_cols):
    scaler = StandardScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
    return dataframe
