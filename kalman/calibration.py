"""Calibracion de la inicializacion, de R y de Q del filtro 1 (seccion 3.3).

Todo se estima SOLO con la particion de entrenamiento. Las tres decisiones y su
justificacion:

**Inicializacion.** El estado arranca en el VECTOR DE JOHANSEN, no en la
solucion OLS. Con la normalizacion beta_{i*} = 1 y la reconstruccion
beta_j = -theta_j, el estado inicial de las patas no numerarias es

    theta_0 = -beta^J_j   para j distinto de i*

y el intercepto es la media en entrenamiento del residual que deja ese vector:

    c_0 = media( P_{i*,t} - X_t theta_0 )

Arrancar en Johansen y no en OLS importa por una razon concreta: el spread que
el filtro produce en sus primeros dias ES el termino de correccion de error del
VECM, que es el mismo objeto sobre el que se estimo la cointegracion y sobre el
que se calibraron la vida media y los umbrales. Con una inicializacion OLS el
filtro arrancaba en un vector distinto (por ejemplo, el peso de PBF salia en
+0.256 contra +0.595 de Johansen) y el spread operado no coincidia con el ECT
que el reporte afirmaba estar operando.

**R, varianza del ruido de observacion.** Es la varianza en entrenamiento de ese
mismo residual, el que deja el vector de Johansen: literalmente, cuanto no
explica la relacion de equilibrio estimada. Se usa el residual de Johansen y no
el de OLS por consistencia con la inicializacion.

**Sigma_0, covarianza inicial.** Se mantiene la covarianza del estimador OLS.
Mide cuanta incertidumbre tiene una estimacion de estos coeficientes con esta
muestra, que es informacion util independientemente de en que punto se arranque,
y ademas fija la escala de Q.

**Q, covarianza del ruido de proceso.** Se parametriza como Q = delta * Sigma_0
con un UNICO escalar libre delta. Dos ventajas: Q hereda las unidades y la
estructura de correlacion correctas de Sigma_0 (los pesos de activos caros
necesitan ruido de distinta escala que los de activos baratos), y la calibracion
se reduce a buscar un numero en una malla. delta es la razon senal-ruido del
estado: cuanto se permite que los pesos se muevan por dia en unidades de su
propia incertidumbre inicial.

delta se elige por maxima log-verosimilitud del filtro sobre entrenamiento. No
es una tecnica extra: la log-verosimilitud es un subproducto de la propia
recursion de Kalman, se acumula gratis en cada paso y es el criterio estandar
para estimar parametros de un modelo de espacio de estados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from kalman import base


def inicializar_desde_johansen(
    precios: pd.DataFrame,
    tickers: list[str],
    pata_numeraria: str,
    beta_johansen: np.ndarray,
) -> dict:
    """Construye el estado inicial, R y Sigma_0 del filtro 1.

    El estado inicial sale del vector de Johansen; Sigma_0 sale de la regresion
    OLS auxiliar, que solo se usa para medir la incertidumbre de un estimador de
    estos coeficientes con esta muestra.

    Args:
        precios: panel restringido a la particion de entrenamiento.
        tickers: activos del grupo, en el orden del vector beta.
        pata_numeraria: ticker usado como variable dependiente.
        beta_johansen: vector cointegrante normalizado, con 1 en la numeraria.

    Returns:
        Diccionario con 'estado_inicial' (theta y c concatenados),
        'covarianza_inicial', 'R', 'nombres_estado' y 'otros'.
    """
    otros = [ticker for ticker in tickers if ticker != pata_numeraria]
    y = precios[pata_numeraria].to_numpy()
    X_otros = precios[otros].to_numpy()
    X = np.column_stack([X_otros, np.ones(len(precios))])

    # Sigma_0: incertidumbre tipica de estos coeficientes en esta muestra.
    coeficientes_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuales_ols = y - X @ coeficientes_ols
    grados_libertad = len(y) - X.shape[1]
    varianza_ols = float(residuales_ols @ residuales_ols / grados_libertad)
    covarianza_coeficientes = varianza_ols * np.linalg.inv(X.T @ X)

    # Estado inicial: el vector de Johansen. beta_j = -theta_j, asi que
    # theta_0 toma el negativo de las componentes no numerarias.
    indices_otros = [tickers.index(ticker) for ticker in otros]
    theta_inicial = -beta_johansen[indices_otros]
    residual_johansen = y - X_otros @ theta_inicial
    intercepto_inicial = float(np.mean(residual_johansen))
    varianza_johansen = float(np.var(residual_johansen, ddof=len(otros) + 1))

    return {
        "estado_inicial": np.append(theta_inicial, intercepto_inicial),
        "covarianza_inicial": covarianza_coeficientes,
        "R": varianza_johansen,
        "nombres_estado": [f"theta_{t}" for t in otros] + ["intercepto"],
        "otros": otros,
        "R_ols_comparacion": varianza_ols,
        "estado_inicial_ols": coeficientes_ols,
    }


def seleccionar_delta(
    observaciones: np.ndarray,
    matriz_observacion: np.ndarray,
    ajuste: dict,
    malla: tuple[float, ...] = config.MALLA_DELTA_Q,
) -> tuple[float, pd.DataFrame]:
    """Elige delta por maxima log-verosimilitud del filtro en entrenamiento.

    Args:
        observaciones: vector y de entrenamiento.
        matriz_observacion: matriz H de entrenamiento.
        ajuste: salida de `inicializar_desde_johansen`.
        malla: valores de delta a evaluar.

    Returns:
        Tupla `(delta_elegido, tabla)` donde la tabla reporta la
        log-verosimilitud y los diagnosticos de innovaciones de cada delta.
    """
    filas = []
    for delta in malla:
        resultado = base.filtrar(
            observaciones,
            matriz_observacion,
            Q=delta * ajuste["covarianza_inicial"],
            R=ajuste["R"],
            media_inicial=ajuste["estado_inicial"],
            covarianza_inicial=ajuste["covarianza_inicial"],
        )
        normalizadas = resultado["innovacion_normalizada"][config.BURN_IN_DIAS :]
        filas.append(
            {
                "delta": delta,
                "log_verosimilitud": resultado["log_verosimilitud"],
                "var_innovacion_normalizada": float(np.var(normalizadas, ddof=1)),
                "autocorr_lag1": float(
                    np.corrcoef(normalizadas[:-1], normalizadas[1:])[0, 1]
                ),
            }
        )

    tabla = pd.DataFrame(filas)
    delta_elegido = float(tabla.loc[tabla["log_verosimilitud"].idxmax(), "delta"])
    tabla["elegido"] = tabla["delta"] == delta_elegido
    return delta_elegido, tabla
