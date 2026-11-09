"""Filtro de Kalman 2: generacion de senal (seccion 3.4).

Opera sobre el spread S_t que produce el filtro 1 y estima en tiempo real la
media latente mu_t y la varianza latente sigma_t^2, lo que permite calcular el
z-score sin sesgo de look-ahead.

Modelo
------
    mu_t = mu_{t-1} + eta_t,        eta_t ~ N(0, Q2_t)      (nivel local)
    S_t  = mu_t + epsilon_t,        epsilon_t ~ N(0, R2_t)

con la varianza de observacion ADAPTATIVA, R2_t = sigma_hat^2_{t|t-1}, y
Q2_t = lambda * R2_t con lambda constante. Mantener constante la razon
Q2/R2 mantiene constante la ganancia estacionaria del filtro, de modo que su
velocidad de adaptacion no depende de la escala del spread. Esto es necesario
porque se trabaja en niveles de precio y los niveles del universo se
multiplicaron por casi diez entre 2012 y 2026: cualquier varianza constante
calibrada en entrenamiento queda mal escalada en validacion.

La varianza latente se estima con una recursion exponencial sobre los errores
de prediccion del propio filtro:

    sigma_hat^2_{t|t-1} = rho * sigma_hat^2_{t-1|t-2} + (1 - rho) * (S_{t-1} - mu_hat_{t-1|t-2})^2

Observese el indice: la varianza con la que se juzga el dia t usa el error del
dia t-1. Nunca entra S_t.

z-score
-------
    z_t = (S_t - mu_hat_{t|t-1}) / sigma_hat_{t|t-1}

Ambos estadisticos son PREVIOS a la actualizacion del filtro con S_t. Normalizar
con la media posterior usaria la observacion del dia para juzgar la observacion
del dia y sesgaria el z-score hacia cero.

Calibracion
-----------
La ventana efectiva se fija en `MULTIPLO_VENTANA_FILTRO2` veces la vida media
medida en entrenamiento, y de ahi se despeja lambda invirtiendo la relacion
entre la razon Q2/R2 y la ganancia estacionaria. Asi el unico numero libre del
filtro 2 queda atado a una cantidad estimada de los datos (la vida media del
paso 3.2.5) en lugar de elegirse a ojo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import artifacts_io
import config


def ganancia_estacionaria(razon_q_r: float) -> float:
    """Ganancia estacionaria del modelo de nivel local para una razon Q2/R2 dada.

    En estado estacionario la covarianza predicha p satisface
    p = (1 - K)(p + Q2) con K = (p + Q2) / (p + Q2 + R2). Normalizando R2 = 1 y
    lambda = Q2/R2, la solucion cerrada de la ganancia es

        K = (-lambda + sqrt(lambda^2 + 4 lambda)) / 2

    Args:
        razon_q_r: lambda = Q2 / R2, estrictamente positiva.

    Returns:
        Ganancia estacionaria en (0, 1).
    """
    return (-razon_q_r + np.sqrt(razon_q_r**2 + 4.0 * razon_q_r)) / 2.0


def razon_q_r_desde_ventana(vida_media_objetivo: float) -> float:
    """Despeja lambda = Q2/R2 para que el filtro tenga la ventana efectiva pedida.

    Un filtro de nivel local en estado estacionario es una media exponencial con
    factor de olvido (1 - K), cuya vida media es h = ln(0.5) / ln(1 - K). Se
    busca por biseccion el lambda cuya ganancia estacionaria produce la h pedida.

    Args:
        vida_media_objetivo: vida media efectiva deseada, en dias habiles.

    Returns:
        La razon lambda = Q2 / R2 correspondiente.

    Raises:
        ValueError: si la vida media objetivo no es mayor a 1 dia.
    """
    if vida_media_objetivo <= 1.0:
        raise ValueError("La vida media objetivo debe ser mayor a 1 dia habil.")

    def vida_media_de(razon: float) -> float:
        K = ganancia_estacionaria(razon)
        return float(np.log(0.5) / np.log(1.0 - K))

    inferior, superior = 1e-10, 10.0
    for _ in range(200):
        medio = np.sqrt(inferior * superior)
        if vida_media_de(medio) > vida_media_objetivo:
            inferior = medio  # lambda mas grande => filtro mas rapido => h menor
        else:
            superior = medio
    return float(np.sqrt(inferior * superior))


def ejecutar_filtro2(
    spread: pd.Series,
    vida_media_entrenamiento: float,
    varianza_semilla: float,
) -> dict:
    """Corre el filtro 2 sobre el spread y produce el z-score ex-ante.

    Args:
        spread: serie S_t producida por el filtro 1, sobre la muestra completa.
        vida_media_entrenamiento: vida media del spread medida en entrenamiento.
        varianza_semilla: varianza inicial del spread, estimada en entrenamiento.

    Returns:
        Diccionario con 'serie' (DataFrame indexado por fecha), 'calibracion' y
        'diagnostico'.
    """
    ventana_objetivo = config.MULTIPLO_VENTANA_FILTRO2 * vida_media_entrenamiento
    razon = razon_q_r_desde_ventana(ventana_objetivo)
    rho = 1.0 - ganancia_estacionaria(razon)

    valores = spread.to_numpy(dtype=float)
    n = len(valores)

    mu_predicha = np.empty(n)
    var_predicha_mu = np.empty(n)
    sigma2_predicha = np.empty(n)
    varianza_innovacion = np.empty(n)
    ganancias = np.empty(n)
    innovaciones = np.empty(n)
    z = np.empty(n)
    mu_posterior = np.empty(n)
    var_posterior_mu = np.empty(n)

    mu = float(valores[0])
    p = float(varianza_semilla)
    sigma2 = float(varianza_semilla)

    for t in range(n):
        # --- Prediccion: nada del dia t ha entrado todavia ---
        R2 = sigma2
        Q2 = razon * R2
        mu_pred = mu
        p_pred = p + Q2

        mu_predicha[t] = mu_pred
        var_predicha_mu[t] = p_pred
        sigma2_predicha[t] = sigma2

        # --- z-score con estadisticos PREVIOS a la actualizacion ---
        innovacion = float(valores[t] - mu_pred)
        F = p_pred + R2
        innovaciones[t] = innovacion
        varianza_innovacion[t] = F
        z[t] = innovacion / np.sqrt(sigma2)

        # --- Actualizacion con la observacion del dia ---
        K = p_pred / F
        ganancias[t] = K
        mu = mu_pred + K * innovacion
        p = (1.0 - K) * p_pred
        mu_posterior[t] = mu
        var_posterior_mu[t] = p

        # La varianza para el dia t+1 incorpora el error de prediccion de HOY,
        # que ya es informacion disponible al cerrar el dia t.
        sigma2 = rho * sigma2 + (1.0 - rho) * innovacion**2

    serie = pd.DataFrame(
        {
            "spread": valores,
            "mu_predicha": mu_predicha,
            "sigma_predicha": np.sqrt(sigma2_predicha),
            "z": z,
            "ganancia_filtro2": ganancias,
            "innovacion": innovaciones,
            "varianza_innovacion": varianza_innovacion,
            "innovacion_normalizada": innovaciones / np.sqrt(varianza_innovacion),
            "mu_posterior": mu_posterior,
            "var_posterior_mu": var_posterior_mu,
        },
        index=spread.index,
    )
    serie["operable"] = np.arange(n) >= config.BURN_IN_DIAS

    normalizadas = serie.loc[serie["operable"], "innovacion_normalizada"].to_numpy()
    diagnostico = {
        "var_innovacion_normalizada": float(np.var(normalizadas, ddof=1)),
        "autocorr_lag1": float(np.corrcoef(normalizadas[:-1], normalizadas[1:])[0, 1]),
        "z_media": float(serie.loc[serie["operable"], "z"].mean()),
        "z_desviacion": float(serie.loc[serie["operable"], "z"].std()),
    }
    calibracion = {
        "vida_media_entrenamiento": vida_media_entrenamiento,
        "multiplo_ventana": config.MULTIPLO_VENTANA_FILTRO2,
        "ventana_efectiva_objetivo": ventana_objetivo,
        "razon_Q2_R2": razon,
        "ganancia_estacionaria": float(ganancia_estacionaria(razon)),
        "rho_varianza": rho,
        "varianza_semilla": varianza_semilla,
    }
    return {"serie": serie, "calibracion": calibracion, "diagnostico": diagnostico}


def guardar_filtro2(grupo: str, resultado: dict) -> list[str]:
    """Escribe el artifact del filtro 2 de un grupo.

    Args:
        grupo: etiqueta del grupo.
        resultado: salida de `ejecutar_filtro2`.

    Returns:
        Lista de rutas relativas escritas.
    """
    artifacts_io.escribir_serie(resultado["serie"], "kalman", f"filter2_{grupo}.parquet")
    return [f"kalman/filter2_{grupo}.parquet"]
