import random
import numpy as np
import pandas as pd


def generar_caso_de_uso_agrupar_eventos_por_franja():
    n = random.randint(12, 28)
    categorias = ["A", "B", "C"]
    base = pd.Timestamp("2026-01-01")

    fechas = [
        base
        + pd.Timedelta(
            days=random.randint(0, 4),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        for _ in range(n)
    ]

    df = pd.DataFrame(
        {
            "fecha_hora": [f.strftime("%Y-%m-%d %H:%M:%S") for f in fechas],
            "categoria": [random.choice(categorias) for _ in range(n)],
            "monto": np.round(np.random.uniform(10, 500, size=n), 2),
        }
    )

    input_data = {"df": df.copy()}

    output_df = df.copy()
    output_df["fecha_hora"] = pd.to_datetime(output_df["fecha_hora"])

    horas = output_df["fecha_hora"].dt.hour
    output_df["franja"] = np.select(
        [
            horas.between(0, 5),
            horas.between(6, 11),
            horas.between(12, 17),
            horas.between(18, 23),
        ],
        ["madrugada", "mañana", "tarde", "noche"],
        default="noche",
    )

    output_df = (
        output_df.groupby(["franja", "categoria"], as_index=False)
        .agg(total_monto=("monto", "sum"), num_eventos=("monto", "size"))
    )

    orden = ["madrugada", "mañana", "tarde", "noche"]
    output_df["franja"] = pd.Categorical(output_df["franja"], categories=orden, ordered=True)

    output_df = output_df.sort_values(by=["franja", "categoria"]).reset_index(drop=True)
    output_df["total_monto"] = output_df["total_monto"].round(2)

    return input_data, output_df