import random
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def generar_caso_de_uso_evaluar_svr_regresion():
    n_samples = random.randint(120, 240)
    n_features = random.randint(4, 8)
    noise = random.uniform(5.0, 25.0)
    random_state_data = random.randint(0, 10000)

    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 1),
        noise=noise,
        random_state=random_state_data,
    )

    test_size = random.choice([0.2, 0.25, 0.3, 0.35])
    random_state = random.randint(0, 1000)

    input_data = {
        "X": X.copy(),
        "y": y.copy(),
        "test_size": test_size,
        "random_state": random_state,
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = SVR(kernel="rbf")
    modelo.fit(X_train_scaled, y_train)

    y_pred = modelo.predict(X_test_scaled)

    output_data = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }

    return input_data, output_data