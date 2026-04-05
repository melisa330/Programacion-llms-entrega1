import random
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def generar_caso_de_uso_evaluar_adaboost_binario():
    n_samples = random.randint(120, 240)
    n_features = random.randint(5, 10)
    random_state_data = random.randint(0, 10000)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(2, n_features - 2),
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        flip_y=0.03,
        class_sep=random.uniform(0.8, 1.8),
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
        stratify=y,
    )

    modelo = AdaBoostClassifier(n_estimators=50, random_state=random_state)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    output_data = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    return input_data, output_data