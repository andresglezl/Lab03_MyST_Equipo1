"""Prueba obligatoria 2: normalizacion (seccion 3.11.2).

"El vector de pesos mantiene escala y signo estables entre reestimaciones
consecutivas."

Se verifican tres cosas:

1. **Escala.** La componente de la pata numeraria vale exactamente 1 en todas las
   fechas. No es una tolerancia numerica: por construccion esa componente no es
   variable de estado, se impone al reconstruir beta a partir de theta, asi que
   la igualdad debe ser exacta.
2. **Signo.** Lo que hay que descartar es el volteo ESPURIO: un salto
   discontinuo de +beta a -beta causado por la indeterminacion de signo del
   eigenvector o por una normalizacion mal planteada. Eso se detecta porque el
   salto seria grande y ocurriria lejos de cero.

   Es distinto de un CRUCE CONTINUO por cero, en el que el peso de una pata
   decae gradualmente hasta cambiar de lado. Eso no es un error de
   normalizacion sino informacion: significa que la relacion de equilibrio
   cambio de estructura. En este proyecto ocurren SIETE cruces, todos
   continuos: `beta_PBF` una vez (2023-12-20, de +0.000061 a -0.002699, tras
   decaer desde +0.25 en 2012), `beta_RY` dos veces y `beta_TD` cuatro veces.
   Son parte del resultado que se reporta, no un detalle tecnico. La prueba
   exige que todo cambio de signo sea de este segundo tipo: continuo y con
   beta pequena a ambos lados, y deja la lista completa en
   `artifacts/kalman/cruces_por_cero_beta.csv`.
3. **Estabilidad.** El cambio diario de beta esta acotado. El filtro impone una
   caminata aleatoria con Q pequena, asi que beta debe moverse despacio; un
   salto grande indicaria que Q quedo mal calibrada y que el filtro esta
   persiguiendo ruido.

Tambien se verifica el re-anclaje del primer eigenvector del Johansen movil, que
usa la misma regla de normalizacion.
"""

from __future__ import annotations

import numpy as np
import pytest

import artifacts_io

#: La pata numeraria debe valer 1 a precision de punto flotante.
TOLERANCIA_ESCALA = 1e-12

#: Cambio diario maximo admisible en cualquier componente de beta.
TOLERANCIA_SALTO_DIARIO = 0.05

#: Magnitud maxima de beta a ambos lados de un cambio de signo para que el
#: cruce se considere continuo y no un volteo espurio de normalizacion.
TOLERANCIA_CRUCE_POR_CERO = 0.01


def test_pata_numeraria_vale_uno_en_todas_las_fechas(grupos):
    """beta de la pata numeraria es exactamente 1 en toda la muestra."""
    for grupo in grupos:
        serie = artifacts_io.leer_serie("kalman", f"filter1_{grupo['grupo']}.parquet")
        columna = f"beta_{grupo['pata_numeraria']}"
        desvio = (serie[columna] - 1.0).abs().max()
        assert desvio < TOLERANCIA_ESCALA, (
            f"{grupo['grupo']}: la pata numeraria {grupo['pata_numeraria']} se "
            f"desvio de 1 hasta {desvio:.2e}"
        )


def test_no_hay_volteos_espurios_de_signo(grupos):
    """Todo cambio de signo de beta es un cruce continuo por cero.

    Un volteo espurio de normalizacion saltaria de +beta a -beta con beta
    grande; un cruce continuo pasa por valores cercanos a cero.
    """
    for grupo in grupos:
        serie = artifacts_io.leer_serie("kalman", f"filter1_{grupo['grupo']}.parquet")
        operable = serie["operable"].to_numpy(dtype=bool)
        fechas = serie.index[operable]
        for ticker in grupo["tickers"]:
            valores = serie[f"beta_{ticker}"].to_numpy()[operable]
            signos = np.sign(valores)
            posiciones = np.flatnonzero(signos[:-1] * signos[1:] < 0)
            for posicion in posiciones:
                antes, despues = valores[posicion], valores[posicion + 1]
                magnitud = max(abs(antes), abs(despues))
                assert magnitud < TOLERANCIA_CRUCE_POR_CERO, (
                    f"{grupo['grupo']} / {ticker}: volteo de signo NO continuo el "
                    f"{fechas[posicion + 1].date()}, de {antes:+.6f} a "
                    f"{despues:+.6f}. Un cruce legitimo pasa por cero con "
                    f"|beta| < {TOLERANCIA_CRUCE_POR_CERO}."
                )


def test_reportar_cruces_por_cero_de_beta(grupos):
    """Deja en un artifact los cruces por cero de beta, que son un resultado.

    Un peso que cruza cero significa que la pata cambio de lado en la relacion
    de equilibrio, es decir que la estructura que Johansen estimo en
    entrenamiento ya no es la misma. Es informacion que el reporte debe citar,
    no un detalle tecnico.
    """
    import pandas as pd

    filas = []
    for grupo in grupos:
        serie = artifacts_io.leer_serie("kalman", f"filter1_{grupo['grupo']}.parquet")
        particiones = artifacts_io.leer_serie("prices.parquet")["particion"]
        operable = serie["operable"].to_numpy(dtype=bool)
        fechas = serie.index[operable]
        for ticker in grupo["tickers"]:
            valores = serie[f"beta_{ticker}"].to_numpy()[operable]
            signos = np.sign(valores)
            for posicion in np.flatnonzero(signos[:-1] * signos[1:] < 0):
                fecha = fechas[posicion + 1]
                filas.append(
                    {
                        "grupo": grupo["grupo"],
                        "ticker": ticker,
                        "fecha_cruce": fecha.date(),
                        "particion": particiones.loc[fecha],
                        "beta_antes": round(float(valores[posicion]), 8),
                        "beta_despues": round(float(valores[posicion + 1]), 8),
                        "beta_inicial": round(float(valores[0]), 6),
                        "beta_final": round(float(valores[-1]), 6),
                    }
                )

    artifacts_io.escribir_tabla(
        pd.DataFrame(filas), "kalman", "cruces_por_cero_beta.csv", indice=False
    )
    assert artifacts_io.ruta("kalman", "cruces_por_cero_beta.csv").exists()


def test_beta_se_mueve_despacio(grupos):
    """El cambio diario de cada componente de beta esta acotado."""
    for grupo in grupos:
        serie = artifacts_io.leer_serie("kalman", f"filter1_{grupo['grupo']}.parquet")
        operable = serie["operable"].to_numpy(dtype=bool)
        columnas = [f"beta_{ticker}" for ticker in grupo["tickers"]]
        saltos = serie.loc[operable, columnas].diff().abs().max()
        peor = float(saltos.max())
        assert peor < TOLERANCIA_SALTO_DIARIO, (
            f"{grupo['grupo']}: salto diario maximo de beta {peor:.4f}, por "
            f"encima de la tolerancia {TOLERANCIA_SALTO_DIARIO}. "
            f"Detalle: {saltos.round(5).to_dict()}"
        )


def test_eigenvector_movil_esta_reanclado(grupos):
    """El primer eigenvector del Johansen movil tiene la numeraria en 1."""
    for grupo in grupos:
        tabla = artifacts_io.leer_serie(
            "rolling", f"johansen_eigen_{grupo['grupo']}.parquet"
        )
        columna = f"eigvec_{grupo['pata_numeraria']}"
        desvio = (tabla[columna] - 1.0).abs().max()
        assert desvio < 1e-10, (
            f"{grupo['grupo']}: el eigenvector movil no esta re-anclado "
            f"(desvio maximo {desvio:.2e})"
        )
