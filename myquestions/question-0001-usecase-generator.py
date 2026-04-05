import random
import pandas as pd


def generar_caso_de_uso_calcular_racha_maxima_lluvia():
    estaciones = ["Arenal", "Bosque", "Centro", "Delta"]
    filas = []

    for estacion in estaciones:
        n_dias = random.randint(8, 18)
        fecha_inicio = pd.Timestamp("2026-01-01") + pd.Timedelta(days=random.randint(0, 10))

        for i in range(n_dias):
            fecha = fecha_inicio + pd.Timedelta(days=i)

            if random.random() < 0.6:
                lluvia = round(random.uniform(0.1, 50.0), 2)
            else:
                lluvia = 0.0

            filas.append([estacion, fecha.strftime("%Y-%m-%d"), lluvia])

    df = pd.DataFrame(filas, columns=["estacion", "fecha", "lluvia_mm"])

    input_data = {"df": df.copy()}

    temp = df.copy()
    temp["fecha"] = pd.to_datetime(temp["fecha"])
    temp = temp.sort_values(by=["estacion", "fecha"]).reset_index(drop=True)

    resultados = []

    for estacion, grupo in temp.groupby("estacion"):
        lluviosos = grupo[grupo["lluvia_mm"] > 0].copy()

        if lluviosos.empty:
            racha_max = 0
        else:
            lluviosos = lluviosos.sort_values("fecha").reset_index(drop=True)
            diffs = lluviosos["fecha"].diff().dt.days
            bloque = (diffs != 1).cumsum()
            racha_max = int(lluviosos.groupby(bloque).size().max())

        resultados.append([estacion, racha_max])

    output_df = pd.DataFrame(resultados, columns=["estacion", "racha_maxima_lluvia"])
    output_df = output_df.sort_values(
        by=["racha_maxima_lluvia", "estacion"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return input_data, output_df


if __name__ == "__main__":
    i, o = generar_caso_de_uso_calcular_racha_maxima_lluvia()

    print("---- inputs ----")
    for k, v in i.items():
        print(f"\n{k}:\n{v}")

    print("\n---- expected output ----")
    print(o)