import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def entrenar_clasificador(df, target_col):

    X_data = df.drop(columns=[target_col]).values
    y_data = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    split = int(0.8 * len(df))

    X_train = X_scaled[:split]
    X_test = X_scaled[split:]

    y_train = y_data[:split]

    model = LogisticRegression()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return preds