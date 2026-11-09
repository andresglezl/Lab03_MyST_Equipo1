"""Risk Parity entre grupos cointegrados (seccion 3.7, nivel B).

Objetivo: asignar el capital entre grupos de modo que cada uno contribuya por
IGUAL al riesgo del portafolio consolidado.

Definicion de contribucion al riesgo
------------------------------------
Con pesos w y covarianza Sigma, la volatilidad del portafolio es
sigma_p = sqrt(w' Sigma w). Es homogenea de grado 1 en w, asi que por el teorema
de Euler se descompone exactamente:

    sigma_p = sum_i w_i * d(sigma_p)/d(w_i) = sum_i w_i (Sigma w)_i / sigma_p

El sumando i-esimo es la contribucion absoluta al riesgo del grupo i, y su
version relativa es

    RC_i = w_i (Sigma w)_i / (w' Sigma w),      sum_i RC_i = 1

Risk Parity pide RC_i = 1/G para todos los grupos.

Por que no basta con el inverso de la volatilidad
-------------------------------------------------
La asignacion w_i proporcional a 1/sigma_i iguala las contribuciones solo si
todas las correlaciones entre grupos son iguales. Con dos grupos cuya
correlacion no es la misma que supone esa receta, las contribuciones quedan
desbalanceadas. Por eso se resuelve el problema completo y se VERIFICA
numericamente la igualdad, como pide la seccion 3.7.

Metodo de solucion
------------------
Descenso coordinado ciclico. Se recorre un grupo a la vez dejando los demas
fijos y se resuelve exactamente su contribucion objetivo. Para el grupo i, pedir
RC_i = 1/G con la normalizacion w'Sigma w = 1 equivale a la ecuacion cuadratica

    sigma_i^2 w_i^2 + w_i * sum_{j != i} sigma_ij w_j - 1/G = 0

cuya raiz positiva es

    w_i = ( -b_i + sqrt( b_i^2 + 4 sigma_i^2 / G ) ) / (2 sigma_i^2),
    b_i = sum_{j != i} sigma_ij w_j

Se itera sobre los grupos hasta convergencia y al final se normaliza para que
los pesos sumen 1.

Por que esta forma y no la iteracion de punto fijo w_i <- 1/(Sigma w)_i: esa
recursion NO garantiza pesos positivos. Si la covarianza tiene terminos cruzados
negativos, como es el caso aqui (la correlacion entre los dos spreads es
negativa), el producto (Sigma w)_i puede volverse negativo y la iteracion
converge a una solucion con un peso negativo y otro mayor a uno. Eso rompe dos
cosas: deja de ser una asignacion de capital interpretable, y hace que la suma de
valores absolutos de los pesos supere 1, con lo que el nocional bruto
desplegado excede el limite del 80% del capital consolidado. La raiz cuadrada de
la actualizacion de arriba es siempre mayor que |b_i|, asi que w_i sale positivo
por construccion.

Retornos de los spreads
-----------------------
La covarianza se estima con los retornos de los SPREADS, no con los retornos de
la estrategia (convencion C6 del plan de fases):

    r_{g,t} = Delta S_{g,t} / sum_k |beta_{k,t-1}| P_{k,t-1}

Es decir el cambio del spread por unidad de nocional bruto. Se usa el spread y
no el P&L de la estrategia por una razon practica: asi los pesos de Risk Parity
no dependen de los umbrales del z-score, y por tanto el riesgo que se reparte es
el del activo subyacente del grupo y no el de una regla de trading particular.
Todo se estima SOLO con la particion de entrenamiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def retorno_spread(
    spread: pd.Series, betas: pd.DataFrame, precios: pd.DataFrame
) -> pd.Series:
    """Retorno del spread por unidad de nocional bruto.

    Args:
        spread: serie S_t del grupo.
        betas: DataFrame con columnas `beta_{ticker}`.
        precios: panel de precios del grupo.

    Returns:
        Serie de retornos, con 0.0 en la primera fecha.
    """
    tickers = [columna.replace("beta_", "") for columna in betas.columns]
    nocional_bruto = (betas.abs().to_numpy() * precios[tickers].to_numpy()).sum(axis=1)
    denominador = pd.Series(nocional_bruto, index=spread.index).shift(1)
    return (spread.diff() / denominador).fillna(0.0)


def contribuciones_al_riesgo(pesos: np.ndarray, covarianza: np.ndarray) -> np.ndarray:
    """Contribuciones relativas al riesgo, que suman 1.

    Args:
        pesos: vector de pesos de los grupos.
        covarianza: matriz de covarianza de los retornos de los spreads.

    Returns:
        Vector `RC` con RC_i = w_i (Sigma w)_i / (w' Sigma w).
    """
    producto = covarianza @ pesos
    varianza = float(pesos @ producto)
    return pesos * producto / varianza


def resolver_risk_parity(
    covarianza: np.ndarray,
    tolerancia: float = config.TOLERANCIA_RISK_PARITY,
    max_iteraciones: int = config.MAX_ITERACIONES_RISK_PARITY,
) -> tuple[np.ndarray, int]:
    """Resuelve los pesos de Risk Parity por descenso coordinado ciclico.

    Args:
        covarianza: matriz de covarianza, definida positiva.
        tolerancia: cambio maximo en los pesos para declarar convergencia.
        max_iteraciones: tope de iteraciones.

    Returns:
        Tupla `(pesos, iteraciones)`.

    Raises:
        RuntimeError: si no converge dentro del tope de iteraciones.
    """
    n = covarianza.shape[0]
    varianzas = np.diag(covarianza)
    if np.any(varianzas <= 0.0):
        raise RuntimeError("Hay grupos con varianza no positiva en entrenamiento.")

    objetivo = 1.0 / n
    pesos = np.full(n, 1.0 / n) / np.sqrt(float(np.full(n, 1.0 / n) @ covarianza @ np.full(n, 1.0 / n)))

    for iteracion in range(1, max_iteraciones + 1):
        anteriores = pesos.copy()
        for i in range(n):
            cruzado = float(covarianza[i] @ pesos - covarianza[i, i] * pesos[i])
            discriminante = cruzado**2 + 4.0 * varianzas[i] * objetivo
            pesos[i] = (-cruzado + np.sqrt(discriminante)) / (2.0 * varianzas[i])
        if np.max(np.abs(pesos - anteriores)) < tolerancia:
            normalizados = pesos / pesos.sum()
            if np.any(normalizados <= 0.0):
                raise RuntimeError(
                    "Risk Parity convergio a pesos no positivos: "
                    f"{normalizados}. Revise la matriz de covarianza."
                )
            return normalizados, iteracion
        pesos = np.clip(pesos, 1e-16, None)

    raise RuntimeError(
        f"Risk Parity no convergio en {max_iteraciones} iteraciones."
    )


def calcular_asignacion(retornos_entrenamiento: pd.DataFrame) -> dict:
    """Calcula los pesos de Risk Parity y los compara con pesos iguales.

    Args:
        retornos_entrenamiento: DataFrame de retornos de spreads por grupo,
            restringido a la particion de entrenamiento.

    Returns:
        Diccionario con los pesos de Risk Parity, los de pesos iguales, las
        contribuciones al riesgo de ambos, la covarianza y la correlacion.
    """
    grupos = list(retornos_entrenamiento.columns)
    covarianza = retornos_entrenamiento.cov().to_numpy()
    pesos_rp, iteraciones = resolver_risk_parity(covarianza)
    pesos_iguales = np.full(len(grupos), 1.0 / len(grupos))

    rc_rp = contribuciones_al_riesgo(pesos_rp, covarianza)
    rc_iguales = contribuciones_al_riesgo(pesos_iguales, covarianza)

    tabla = pd.DataFrame(
        {
            "grupo": grupos,
            "peso_risk_parity": pesos_rp.round(8),
            "peso_iguales": pesos_iguales.round(8),
            "vol_anual_spread": (
                retornos_entrenamiento.std(ddof=1).to_numpy()
                * np.sqrt(config.DIAS_ANUALIZACION)
            ).round(6),
            "contribucion_riesgo_rp": rc_rp.round(10),
            "contribucion_riesgo_iguales": rc_iguales.round(10),
        }
    )
    tabla["objetivo_contribucion"] = 1.0 / len(grupos)
    tabla["desvio_rp"] = (tabla["contribucion_riesgo_rp"] - 1.0 / len(grupos)).abs()

    return {
        "grupos": grupos,
        "pesos": dict(zip(grupos, pesos_rp)),
        "pesos_iguales": dict(zip(grupos, pesos_iguales)),
        "covarianza": pd.DataFrame(covarianza, index=grupos, columns=grupos),
        "correlacion": retornos_entrenamiento.corr(),
        "tabla": tabla,
        "iteraciones": iteraciones,
        "desvio_maximo": float(tabla["desvio_rp"].max()),
        "tolerancia": config.TOLERANCIA_RISK_PARITY,
        "contribuciones_iguales": bool(
            tabla["desvio_rp"].max() < config.TOLERANCIA_RISK_PARITY
        ),
    }
