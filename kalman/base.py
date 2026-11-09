"""Recursion generica del filtro de Kalman lineal-gaussiano con observacion escalar.

Los dos filtros del laboratorio son casos particulares del mismo modelo:

    Transicion:   x_t = x_{t-1} + w_t,      w_t ~ N(0, Q)
    Observacion:  y_t = H_t x_t + v_t,      v_t ~ N(0, R)

El filtro 1 tiene estado x_t = (theta_t, c_t) y H_t = (P_{-i*,t}, 1), que cambia
cada dia porque contiene precios. El filtro 2 tiene estado escalar x_t = mu_t y
H_t = (1), constante. Compartir la recursion evita duplicar codigo y garantiza
que la ganancia y las innovaciones se calculen igual en ambos, que es lo que la
prueba obligatoria 4 verifica.

Ambas matrices de transicion son la identidad: es la hipotesis de CAMINATA
ALEATORIA sobre el estado. Es deliberada y se defiende asi: no tenemos un modelo
de como deberian moverse los pesos de cointegracion ni la media del spread, solo
sabemos que se mueven despacio. La caminata aleatoria es la transicion que menos
estructura impone, y toda la velocidad de adaptacion queda concentrada en un
unico parametro, Q, que se calibra con datos de entrenamiento.
"""

from __future__ import annotations

import numpy as np


def paso_prediccion(
    media: np.ndarray, covarianza: np.ndarray, Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Propaga el estado un periodo hacia adelante (ecuaciones de prediccion).

    Con transicion identidad, la media no cambia y la covarianza crece en Q:
    el filtro se vuelve mas incierto a medida que pasa el tiempo sin observar.

    Args:
        media: estado posterior del periodo anterior, x_{t-1|t-1}.
        covarianza: covarianza posterior del periodo anterior, Sigma_{t-1|t-1}.
        Q: covarianza del ruido de proceso.

    Returns:
        Tupla `(x_{t|t-1}, Sigma_{t|t-1})`.
    """
    return media.copy(), covarianza + Q


def paso_actualizacion(
    media_predicha: np.ndarray,
    covarianza_predicha: np.ndarray,
    observacion: float,
    H: np.ndarray,
    R: float,
) -> dict:
    """Incorpora la observacion del dia (ecuaciones de actualizacion).

    La ganancia de Kalman es K_t = Sigma_{t|t-1} H_t' / F_t, donde
    F_t = H_t Sigma_{t|t-1} H_t' + R es la varianza de la innovacion. Su
    interpretacion es un cociente de incertidumbres: el numerador es cuanta
    duda tiene el filtro sobre su propio estado y el denominador cuanta duda
    total hay sumando el ruido de observacion. Si R es grande frente a
    Sigma_{t|t-1}, K_t tiende a cero y el filtro ignora la observacion; si es
    pequeno, K_t crece y el filtro se mueve casi todo lo que la innovacion
    sugiere.

    Args:
        media_predicha: x_{t|t-1}.
        covarianza_predicha: Sigma_{t|t-1}.
        observacion: y_t.
        H: fila de observacion H_t, de forma (n,).
        R: varianza del ruido de observacion.

    Returns:
        Diccionario con el estado posterior, la covarianza posterior, la
        innovacion, su varianza, la ganancia, la innovacion normalizada y la
        contribucion a la log-verosimilitud.
    """
    innovacion = float(observacion - H @ media_predicha)
    varianza_innovacion = float(H @ covarianza_predicha @ H + R)
    ganancia = covarianza_predicha @ H / varianza_innovacion

    media_posterior = media_predicha + ganancia * innovacion
    n = len(media_predicha)
    covarianza_posterior = (np.eye(n) - np.outer(ganancia, H)) @ covarianza_predicha
    # Se simetriza para evitar que el error de redondeo acumulado haga perder la
    # simetria de la covarianza a lo largo de miles de iteraciones.
    covarianza_posterior = 0.5 * (covarianza_posterior + covarianza_posterior.T)

    log_verosimilitud = -0.5 * (
        np.log(2.0 * np.pi * varianza_innovacion) + innovacion**2 / varianza_innovacion
    )

    return {
        "media_posterior": media_posterior,
        "covarianza_posterior": covarianza_posterior,
        "innovacion": innovacion,
        "varianza_innovacion": varianza_innovacion,
        "ganancia": ganancia,
        "innovacion_normalizada": innovacion / np.sqrt(varianza_innovacion),
        "log_verosimilitud": float(log_verosimilitud),
    }


def filtrar(
    observaciones: np.ndarray,
    matriz_observacion: np.ndarray,
    Q: np.ndarray,
    R: float,
    media_inicial: np.ndarray,
    covarianza_inicial: np.ndarray,
) -> dict:
    """Corre la recursion completa sobre una muestra.

    El recorrido es estrictamente causal: en el periodo t solo se usan las
    observaciones hasta t. Esa es la propiedad que hace que el z-score no tenga
    look-ahead y la que verifica la prueba obligatoria 1.

    Args:
        observaciones: vector y de forma (T,).
        matriz_observacion: matriz H de forma (T, n); la fila t es H_t.
        Q: covarianza del ruido de proceso, (n, n).
        R: varianza del ruido de observacion.
        media_inicial: x_{0|0}.
        covarianza_inicial: Sigma_{0|0}.

    Returns:
        Diccionario de arrays con el estado predicho y posterior, las diagonales
        de las covarianzas, la innovacion y su varianza, la ganancia, la
        innovacion normalizada y la log-verosimilitud total.
    """
    n_periodos, n_estado = matriz_observacion.shape
    estado_predicho = np.empty((n_periodos, n_estado))
    estado_posterior = np.empty((n_periodos, n_estado))
    varianza_predicha = np.empty((n_periodos, n_estado))
    varianza_posterior = np.empty((n_periodos, n_estado))
    ganancias = np.empty((n_periodos, n_estado))
    innovaciones = np.empty(n_periodos)
    varianzas_innovacion = np.empty(n_periodos)
    innovaciones_normalizadas = np.empty(n_periodos)

    media, covarianza = media_inicial.astype(float), covarianza_inicial.astype(float)
    log_verosimilitud_total = 0.0

    for t in range(n_periodos):
        media_pred, covarianza_pred = paso_prediccion(media, covarianza, Q)
        estado_predicho[t] = media_pred
        varianza_predicha[t] = np.diag(covarianza_pred)

        resultado = paso_actualizacion(
            media_pred, covarianza_pred, observaciones[t], matriz_observacion[t], R
        )
        media, covarianza = resultado["media_posterior"], resultado["covarianza_posterior"]

        estado_posterior[t] = media
        varianza_posterior[t] = np.diag(covarianza)
        ganancias[t] = resultado["ganancia"]
        innovaciones[t] = resultado["innovacion"]
        varianzas_innovacion[t] = resultado["varianza_innovacion"]
        innovaciones_normalizadas[t] = resultado["innovacion_normalizada"]
        log_verosimilitud_total += resultado["log_verosimilitud"]

    return {
        "estado_predicho": estado_predicho,
        "estado_posterior": estado_posterior,
        "varianza_predicha": varianza_predicha,
        "varianza_posterior": varianza_posterior,
        "ganancia": ganancias,
        "innovacion": innovaciones,
        "varianza_innovacion": varianzas_innovacion,
        "innovacion_normalizada": innovaciones_normalizadas,
        "log_verosimilitud": log_verosimilitud_total,
    }
