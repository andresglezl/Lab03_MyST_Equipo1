"""Ejemplo numerico trabajado de ambos filtros (seccion 3.5).

La seccion 3.5 exige "un ejemplo numerico trabajado que muestre la evolucion del
estado a lo largo de varios periodos". Este modulo lo produce con los datos y
los parametros REALES del proyecto, no con numeros inventados, de modo que el
lector pueda seguir la aritmetica paso a paso y verificarla contra los artifacts.

Se toman los primeros periodos inmediatamente posteriores al burn-in y se
tabulan, para cada periodo, todas las cantidades de la recursion:

  - El estado predicho x_{t|t-1} y su covarianza Sigma_{t|t-1}.
  - La observacion y_t y la fila de observacion H_t.
  - La innovacion nu_t = y_t - H_t x_{t|t-1} y su varianza F_t.
  - La ganancia K_t = Sigma_{t|t-1} H_t' / F_t.
  - El estado posterior x_{t|t} = x_{t|t-1} + K_t nu_t.

Asi el ejemplo no es ilustrativo sino auditable: cada numero sale del mismo
codigo que produce el backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import artifacts_io
import config
from kalman import base, calibration, filter1_weights

#: Numero de periodos que se tabulan en el ejemplo.
N_PERIODOS = 6


def ejemplo_filtro1(
    precios: pd.DataFrame,
    precios_entrenamiento: pd.DataFrame,
    tickers: list[str],
    pata_numeraria: str,
    beta_johansen: np.ndarray,
    inicio: int = config.BURN_IN_DIAS,
    n_periodos: int = N_PERIODOS,
) -> pd.DataFrame:
    """Tabula la recursion del filtro 1 periodo por periodo.

    Args:
        precios: panel completo del grupo.
        precios_entrenamiento: panel del grupo en entrenamiento.
        tickers: activos del grupo en el orden del vector beta.
        pata_numeraria: ticker numerario.
        beta_johansen: vector cointegrante que fija el estado inicial.
        inicio: indice del primer periodo a tabular.
        n_periodos: cuantos periodos tabular.

    Returns:
        DataFrame con una fila por periodo y todas las cantidades de la recursion.
    """
    ajuste = calibration.inicializar_desde_johansen(
        precios_entrenamiento, tickers, pata_numeraria, beta_johansen
    )
    y_entrenamiento, H_entrenamiento, _ = filter1_weights.preparar_observaciones(
        precios_entrenamiento, tickers, pata_numeraria
    )
    delta, _ = calibration.seleccionar_delta(y_entrenamiento, H_entrenamiento, ajuste)
    Q = delta * ajuste["covarianza_inicial"]
    y, H, otros = filter1_weights.preparar_observaciones(
        precios, tickers, pata_numeraria
    )

    media = ajuste["estado_inicial"].astype(float)
    covarianza = ajuste["covarianza_inicial"].astype(float)
    nombres = ajuste["nombres_estado"]

    filas = []
    for t in range(inicio + n_periodos):
        media_pred, covarianza_pred = base.paso_prediccion(media, covarianza, Q)
        paso = base.paso_actualizacion(
            media_pred, covarianza_pred, y[t], H[t], ajuste["R"]
        )
        if t >= inicio:
            fila = {
                "periodo": t,
                "fecha": precios.index[t],
                f"observacion_y_({pata_numeraria})": y[t],
            }
            fila.update({f"H_{nombre}": H[t][i] for i, nombre in enumerate(nombres)})
            fila.update(
                {f"pred_{nombre}": media_pred[i] for i, nombre in enumerate(nombres)}
            )
            fila["innovacion_nu"] = paso["innovacion"]
            fila["var_innovacion_F"] = paso["varianza_innovacion"]
            fila["innovacion_normalizada"] = paso["innovacion_normalizada"]
            fila.update(
                {
                    f"ganancia_{nombre}": paso["ganancia"][i]
                    for i, nombre in enumerate(nombres)
                }
            )
            fila.update(
                {
                    f"post_{nombre}": paso["media_posterior"][i]
                    for i, nombre in enumerate(nombres)
                }
            )
            # El spread operable es la innovacion: usa el estado PREDICHO.
            fila["spread_S"] = paso["innovacion"]
            for posicion, ticker in enumerate(otros):
                fila[f"beta_{ticker}"] = -media_pred[posicion]
            fila[f"beta_{pata_numeraria}"] = 1.0
            filas.append(fila)

        media = paso["media_posterior"]
        covarianza = paso["covarianza_posterior"]

    return pd.DataFrame(filas).set_index("periodo")


def ejemplo_filtro2(
    spread: pd.Series,
    vida_media: float,
    varianza_semilla: float,
    inicio: int = config.BURN_IN_DIAS,
    n_periodos: int = N_PERIODOS,
) -> pd.DataFrame:
    """Tabula la recursion del filtro 2 periodo por periodo.

    Args:
        spread: serie S_t producida por el filtro 1.
        vida_media: vida media del spread estimada en entrenamiento.
        varianza_semilla: varianza inicial del spread.
        inicio: indice del primer periodo a tabular.
        n_periodos: cuantos periodos tabular.

    Returns:
        DataFrame con una fila por periodo mostrando mu, sigma, z, K y el
        posterior.
    """
    from kalman import filter2_signal

    razon = filter2_signal.razon_q_r_desde_ventana(
        config.MULTIPLO_VENTANA_FILTRO2 * vida_media
    )
    rho = 1.0 - filter2_signal.ganancia_estacionaria(razon)

    valores = spread.to_numpy(dtype=float)
    mu, p, sigma2 = float(valores[0]), float(varianza_semilla), float(varianza_semilla)

    filas = []
    for t in range(inicio + n_periodos):
        R2 = sigma2
        Q2 = razon * R2
        mu_pred, p_pred = mu, p + Q2
        innovacion = float(valores[t] - mu_pred)
        F = p_pred + R2
        z = innovacion / np.sqrt(sigma2)
        K = p_pred / F

        if t >= inicio:
            filas.append(
                {
                    "periodo": t,
                    "fecha": spread.index[t],
                    "observacion_S": valores[t],
                    "mu_predicha": mu_pred,
                    "var_predicha_p": p_pred,
                    "sigma_predicha": np.sqrt(sigma2),
                    "R2_adaptativa": R2,
                    "Q2": Q2,
                    "innovacion_nu": innovacion,
                    "var_innovacion_F": F,
                    "z_score": z,
                    "ganancia_K": K,
                    "mu_posterior": mu_pred + K * innovacion,
                    "var_posterior": (1.0 - K) * p_pred,
                }
            )

        mu = mu_pred + K * innovacion
        p = (1.0 - K) * p_pred
        sigma2 = rho * sigma2 + (1.0 - rho) * innovacion**2

    return pd.DataFrame(filas).set_index("periodo")


def generar_y_guardar(
    panel: pd.DataFrame,
    etiquetas: pd.Series,
    resultados_grupo: list[dict],
) -> list[str]:
    """Produce los ejemplos numericos de todos los grupos y los escribe.

    Args:
        panel: panel de precios limpio.
        etiquetas: serie de particiones por fecha.
        resultados_grupo: salidas de `strategy.zscore.procesar_grupo`.

    Returns:
        Lista de rutas relativas escritas en `artifacts/`.
    """
    mascara = (etiquetas == "entrenamiento").to_numpy()
    salidas = []

    for resultado in resultados_grupo:
        grupo = resultado["grupo"]
        tickers = resultado["tickers"]
        numeraria = resultado["resumen"]["pata_numeraria"]
        vida_media = resultado["resumen"]["vecm"]["vida_media_dias"]

        tabla1 = ejemplo_filtro1(
            panel[tickers],
            panel[tickers].loc[mascara],
            tickers,
            numeraria,
            resultado["beta_johansen"],
        )
        spread = resultado["serie_f1"]["spread"]
        operable = resultado["serie_f1"]["operable"]
        semilla = float(spread.loc[mascara & operable.to_numpy()].var(ddof=1))
        tabla2 = ejemplo_filtro2(spread, vida_media, semilla)

        artifacts_io.escribir_tabla(
            tabla1.round(8), "kalman", f"worked_example_filtro1_{grupo}.csv"
        )
        artifacts_io.escribir_tabla(
            tabla2.round(8), "kalman", f"worked_example_filtro2_{grupo}.csv"
        )
        salidas += [
            f"kalman/worked_example_filtro1_{grupo}.csv",
            f"kalman/worked_example_filtro2_{grupo}.csv",
        ]

    return salidas
