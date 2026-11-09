"""Johansen movil: primer eigenvector a lo largo del tiempo (figura 4).

Esta pieza no genera senales; es un DIAGNOSTICO de estabilidad estructural. El
filtro 1 estima beta_t de forma recursiva imponiendo una caminata aleatoria;
este modulo reestima Johansen desde cero en ventanas traslapadas, sin imponer
nada. Comparar las dos trayectorias responde una pregunta que el profesor puede
hacer: el filtro se mueve porque la relacion de equilibrio realmente cambio, o
porque Q lo deja perseguir ruido.

Convencion de signo: cada ventana produce un eigenvector con escala y signo
indeterminados, asi que se re-ancla con la MISMA regla del plan de fases (C2),
dividiendo por la componente de la pata numeraria congelada en entrenamiento.
Sin este re-anclaje la figura mostraria volteos de signo espurios entre
ventanas consecutivas que no significan nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import artifacts_io
import config
from cointegration import johansen


def johansen_movil(
    precios: pd.DataFrame,
    tickers: list[str],
    pata_numeraria: str,
    ventana: int = config.VENTANA_JOHANSEN_MOVIL,
    paso: int = config.PASO_JOHANSEN_MOVIL,
) -> pd.DataFrame:
    """Reestima Johansen en ventanas traslapadas y devuelve el primer eigenvector.

    Args:
        precios: panel completo del grupo.
        tickers: activos del grupo en el orden del vector beta.
        pata_numeraria: ticker numerario congelado desde entrenamiento.
        ventana: tamano de la ventana en dias habiles.
        paso: cada cuantos dias se reestima.

    Returns:
        DataFrame indexado por la fecha de cierre de cada ventana, con los
        eigenvalores, el estadistico de traza, el rango estimado y las
        componentes del primer eigenvector re-ancladas.
    """
    indice_numerario = tickers.index(pata_numeraria)
    filas = []

    for fin in range(ventana, len(precios) + 1, paso):
        bloque = precios.iloc[fin - ventana : fin]
        resultado = johansen.prueba_johansen(bloque, tuple(tickers))

        vector = resultado["eigenvectores"][:, 0]
        pivote = vector[indice_numerario]
        if np.isclose(pivote, 0.0):
            continue
        vector_anclado = vector / pivote

        filas.append(
            {
                "fecha": bloque.index[-1],
                "eigenvalor_1": float(resultado["eigenvalores"][0]),
                "traza_r0": resultado["traza_r0"],
                "traza_critico_95": resultado["traza_critico_r0"],
                "cointegra_95": resultado["pasa_johansen"],
                "rango": resultado["rango"],
                **{
                    f"eigvec_{ticker}": float(valor)
                    for ticker, valor in zip(tickers, vector_anclado)
                },
            }
        )

    return pd.DataFrame(filas).set_index("fecha")


def guardar_johansen_movil(grupo: str, tabla: pd.DataFrame) -> list[str]:
    """Escribe el artifact del Johansen movil de un grupo.

    Args:
        grupo: etiqueta del grupo.
        tabla: salida de `johansen_movil`.

    Returns:
        Lista de rutas relativas escritas.
    """
    artifacts_io.escribir_serie(tabla, "rolling", f"johansen_eigen_{grupo}.parquet")
    return [f"rolling/johansen_eigen_{grupo}.parquet"]
