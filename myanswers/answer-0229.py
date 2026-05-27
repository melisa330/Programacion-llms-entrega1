import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error

def entrenar_modelo_estacional(df, target_col, fecha_col):

    df_proc = df.copy()

    df_proc[fecha_col] = pd.to_datetime(df_proc[fecha_col])

    day = df_proc[fecha_col].dt.dayofweek

    df_proc['dia_sin'] = np.sin(2 * np.pi * day / 7)
    df_proc['dia_cos'] = np.cos(2 * np.pi * day / 7)

    X = df_proc.drop(columns=[fecha_col, target_col])
    y = df_proc[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Lasso(alpha=0.1)

    model.fit(X_scaled, y)

    coef_ceros = np.sum(
        np.isclose(model.coef_, 0, atol=1e-5)
    )

    mae = mean_absolute_error(
        y,
        model.predict(X_scaled)
    )

    return (model, int(coef_ceros), float(mae))