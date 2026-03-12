import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



def load_data():
    df = pd.read_csv("../Data/loan_prediction_dataset.csv")
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.columns)

    le = LabelEncoder()
    df['Employment_Status'] = le.fit_transform(df['Employment_Status'])

    return df