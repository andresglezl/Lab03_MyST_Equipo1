"""Filtro de Kalman 1: pesos dinamicos del portafolio (seccion 3.3).

Formulacion (convencion C1 del plan de fases). Con pata numeraria i* fija y
tilde{P}_t los precios de las demas patas, el estado es x_t = (theta_t, c_t):

    x_t = x_{t-1} + w_t,                        w_t ~ N(0, Q)
    y_t = P_{i*,t} = H_t x_t + v_t,             H_t = (tilde{P}_t', 1),  v_t ~ N(0, R)

El vector de pesos se reconstruye con beta_{i*,t} = 1 y beta_{j,t} = -theta_{j,t},
de modo que la componente numeraria vale +1 por construccion y no es variable de
estado. Eso es lo que hace que la prueba obligatoria 2 se cumpla exactamente y
no por tolerancia.

El spread operable es la INNOVACION, es decir usa el estado PREDICHO y nunca el
posterior:

    S_t = beta_{t|t-1}' P_t - c_{t|t-1} = y_t - H_t x_{t|t-1} = nu_t

Usar el estado posterior equivaldria a usar la observacion del dia para juzgar
la observacion del dia, lo que haria que el spread pareciera mas pequeno de lo
que es y sesgaria toda la senal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import artifacts_io
import config
from kalman import base, calibration


def preparar_observaciones(
    precios: pd.DataFrame, tickers: list[str], pata_numeraria: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construye el vector de observaciones y y la matriz de observacion H.

    Args:
        precios: panel de precios del grupo.
        tickers: activos del grupo en el orden del vector beta.
        pata_numeraria: ticker que va al lado izquierdo de la ecuacion.

    Returns:
        Tupla `(y, H, otros)` donde `otros` son los tickers no numerarios en el
        orden en que aparecen en las columnas de H.
    """
    otros = [ticker for ticker in tickers if ticker != pata_numeraria]
    y = precios[pata_numeraria].to_numpy()
    H = np.column_stack([precios[otros].to_numpy(), np.ones(len(precios))])
    return y, H, otros


def reconstruir_betas(
    thetas: np.ndarray, tickers: list[str], otros: list[str], pata_numeraria: str
) -> pd.DataFrame:
    """Convierte el estado theta en el vector de pesos beta completo.

    Args:
        thetas: matriz (T, n_otros + 1); la ultima columna es el intercepto.
        tickers: activos del grupo en el orden del vector beta.
        otros: tickers no numerarios en el orden de las columnas de theta.
        pata_numeraria: ticker numerario.

    Returns:
        DataFrame con una columna `beta_{ticker}` por activo del grupo.
    """
    betas = pd.DataFrame(index=range(len(thetas)))
    for posicion, ticker in enumerate(otros):
        betas[f"beta_{ticker}"] = -thetas[:, posicion]
    betas[f"beta_{pata_numeraria}"] = 1.0
    return betas[[f"beta_{ticker}" for ticker in tickers]]


def ejecutar_filtro1(
    precios: pd.DataFrame,
    precios_entrenamiento: pd.DataFrame,
    tickers: list[str],
    pata_numeraria: str,
    beta_johansen: np.ndarray,
) -> dict:
    """Calibra y corre el filtro 1 sobre la muestra completa.

    La calibracion de Q, R y el estado inicial usa SOLO entrenamiento; la
    recursion recorre toda la muestra, lo que es legitimo porque es causal:
    en el dia t solo usa precios hasta t.

    Args:
        precios: panel completo del grupo (los tres tramos).
        precios_entrenamiento: panel del grupo restringido a entrenamiento.
        tickers: activos del grupo en el orden del vector beta.
        pata_numeraria: ticker numerario, congelado desde entrenamiento.
        beta_johansen: vector cointegrante normalizado de entrenamiento, que
            fija el estado inicial del filtro.

    Returns:
        Diccionario con 'serie' (DataFrame indexado por fecha), 'calibracion'
        (parametros y tabla de la malla de delta) y 'diagnostico'.
    """
    ajuste = calibration.inicializar_desde_johansen(
        precios_entrenamiento, tickers, pata_numeraria, beta_johansen
    )
    y_entrenamiento, H_entrenamiento, _ = preparar_observaciones(
        precios_entrenamiento, tickers, pata_numeraria
    )
    delta, tabla_delta = calibration.seleccionar_delta(
        y_entrenamiento, H_entrenamiento, ajuste
    )

    Q = delta * ajuste["covarianza_inicial"]
    y, H, otros = preparar_observaciones(precios, tickers, pata_numeraria)
    resultado = base.filtrar(
        y,
        H,
        Q=Q,
        R=ajuste["R"],
        media_inicial=ajuste["estado_inicial"],
        covarianza_inicial=ajuste["covarianza_inicial"],
    )

    betas_predichas = reconstruir_betas(
        resultado["estado_predicho"], tickers, otros, pata_numeraria
    )
    betas_predichas.index = precios.index

    serie = pd.DataFrame(index=precios.index)
    for columna in betas_predichas.columns:
        serie[columna] = betas_predichas[columna]
    serie["intercepto"] = resultado["estado_predicho"][:, -1]
    serie["spread"] = resultado["innovacion"]
    serie["varianza_innovacion"] = resultado["varianza_innovacion"]
    serie["innovacion_normalizada"] = resultado["innovacion_normalizada"]
    for posicion, nombre in enumerate(ajuste["nombres_estado"]):
        serie[f"ganancia_{nombre}"] = resultado["ganancia"][:, posicion]
        serie[f"var_post_{nombre}"] = resultado["varianza_posterior"][:, posicion]
    serie["ganancia_norma"] = np.linalg.norm(resultado["ganancia"], axis=1)
    serie["traza_var_posterior"] = resultado["varianza_posterior"].sum(axis=1)
    serie["operable"] = np.arange(len(serie)) >= config.BURN_IN_DIAS

    normalizadas = resultado["innovacion_normalizada"][config.BURN_IN_DIAS :]
    diagnostico = {
        "var_innovacion_normalizada": float(np.var(normalizadas, ddof=1)),
        "autocorr_lag1": float(np.corrcoef(normalizadas[:-1], normalizadas[1:])[0, 1]),
        "log_verosimilitud": resultado["log_verosimilitud"],
    }

    calibracion = {
        "pata_numeraria": pata_numeraria,
        "nombres_estado": ajuste["nombres_estado"],
        "R": ajuste["R"],
        "R_ols_comparacion": ajuste["R_ols_comparacion"],
        "estado_inicial_johansen": ajuste["estado_inicial"].tolist(),
        "delta_elegido": delta,
        "Q_diagonal": np.diag(Q).tolist(),
        "estado_inicial": ajuste["estado_inicial"].tolist(),
        "burn_in_dias": config.BURN_IN_DIAS,
        "tabla_delta": tabla_delta,
    }
    return {"serie": serie, "calibracion": calibracion, "diagnostico": diagnostico}


def guardar_filtro1(grupo: str, resultado: dict) -> list[str]:
    """Escribe los artifacts del filtro 1 de un grupo.

    Args:
        grupo: etiqueta del grupo, p. ej. 'VLO-MPC-PSX-PBF'.
        resultado: salida de `ejecutar_filtro1`.

    Returns:
        Lista de rutas relativas escritas.
    """
    artifacts_io.escribir_serie(resultado["serie"], "kalman", f"filter1_{grupo}.parquet")
    artifacts_io.escribir_tabla(
        resultado["calibracion"]["tabla_delta"],
        "kalman",
        f"calibracion_delta_{grupo}.csv",
        indice=False,
    )
    return [
        f"kalman/filter1_{grupo}.parquet",
        f"kalman/calibracion_delta_{grupo}.csv",
    ]
