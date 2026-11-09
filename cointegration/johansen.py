"""Paso 4 del embudo: prueba de Johansen (seccion 3.2.4).

Johansen es el criterio DEFINITIVO del embudo porque, a diferencia de
Engle-Granger, no exige elegir una variable dependiente y permite detectar mas
de una relacion de cointegracion entre los cuatro activos.

Especificacion declarada (`config.JOHANSEN_*`):
  - `det_order=0`: constante NO RESTRINGIDA. En statsmodels este valor mete una
    constante al VAR sin restringirla a la relacion de cointegracion, asi que
    el spread beta'P no se fuerza a tener media cero. Su nivel lo absorben el
    intercepto `c_t` del filtro 1 y la media latente `mu_t` del filtro 2.
  - `k_ar_diff=1`: un rezago en diferencias, por parsimonia. Se verifico que la
    conclusion del test no cambia con k en {1, 2, 5} ni con k elegido por AIC,
    de modo que el resultado no es un artefacto de la seleccion de rezagos.
  - Nivel de significancia 95% (`JOHANSEN_NIVEL_CRITICO=1`).

El rango de cointegracion se determina con el estadistico de TRAZA por el
procedimiento secuencial: se empieza en r=0 y se avanza mientras el estadistico
supere su valor critico. El primer r que no lo supere es el rango estimado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import config


def prueba_johansen(precios: pd.DataFrame, grupo: tuple[str, ...]) -> dict:
    """Aplica Johansen a un grupo y devuelve traza, max-eigenvalor y rango.

    Args:
        precios: panel de precios restringido a la particion de entrenamiento.
        grupo: tickers del grupo candidato.

    Returns:
        Diccionario con los estadisticos de traza y de maximo eigenvalor, sus
        valores criticos al 95%, el rango estimado, los eigenvalores y la matriz
        de eigenvectores (una columna por relacion de cointegracion).
    """
    activos = list(grupo)
    resultado = coint_johansen(
        precios[activos].to_numpy(),
        det_order=config.JOHANSEN_DET_ORDER,
        k_ar_diff=config.JOHANSEN_K_AR_DIFF,
    )

    nivel = config.JOHANSEN_NIVEL_CRITICO
    traza = resultado.lr1
    traza_critico = resultado.cvt[:, nivel]
    maxeig = resultado.lr2
    maxeig_critico = resultado.cvm[:, nivel]

    rango = 0
    while rango < len(traza) and traza[rango] > traza_critico[rango]:
        rango += 1

    return {
        "tickers": activos,
        "traza": traza,
        "traza_critico_95": traza_critico,
        "maxeig": maxeig,
        "maxeig_critico_95": maxeig_critico,
        "eigenvalores": resultado.eig,
        "eigenvectores": resultado.evec,
        "rango": int(rango),
        "traza_r0": float(traza[0]),
        "traza_critico_r0": float(traza_critico[0]),
        "maxeig_r0": float(maxeig[0]),
        "maxeig_critico_r0": float(maxeig_critico[0]),
        "pasa_johansen": bool(rango >= 1),
    }


def resumen_johansen_para_artifact(resultado: dict) -> dict:
    """Convierte el resultado de Johansen en un diccionario serializable a JSON.

    Args:
        resultado: salida de `prueba_johansen`.

    Returns:
        Diccionario con arrays convertidos a listas de floats redondeados.
    """
    return {
        "tickers": resultado["tickers"],
        "especificacion": {
            "det_order": config.JOHANSEN_DET_ORDER,
            "k_ar_diff": config.JOHANSEN_K_AR_DIFF,
            "nivel_significancia": "95%",
        },
        "estadistico_traza": np.round(resultado["traza"], 4).tolist(),
        "traza_critico_95": np.round(resultado["traza_critico_95"], 4).tolist(),
        "estadistico_maxeig": np.round(resultado["maxeig"], 4).tolist(),
        "maxeig_critico_95": np.round(resultado["maxeig_critico_95"], 4).tolist(),
        "eigenvalores": np.round(resultado["eigenvalores"], 8).tolist(),
        "eigenvectores": np.round(resultado["eigenvectores"], 8).tolist(),
        "rango_estimado": resultado["rango"],
    }
