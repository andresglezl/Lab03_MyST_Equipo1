"""Metricas de desempeno (seccion 3.8-Dashboard, tarjeta de desempeno).

Todas las metricas de retorno se anualizan con 252 dias habiles
(`config.DIAS_ANUALIZACION`) y con tasa libre de riesgo cero. La tasa cero se
declara como supuesto: el objetivo es comparar la estrategia contra si misma en
los tres tramos y contra las asignaciones alternativas, no contra un bono.

Definiciones usadas, para que sean verificables:

    Sharpe  = media(r) / desv(r) * sqrt(252)
    Sortino = media(r) / sqrt( media( min(r, 0)^2 ) ) * sqrt(252)
    Calmar  = CAGR / |max drawdown|
    MaxDD   = min_t ( C_t / max_{s<=t} C_s - 1 )
    WinRate = operaciones con P&L neto positivo / operaciones cerradas
    ProfitFactor = suma de P&L de ganadoras / |suma de P&L de perdedoras|

El denominador del Sortino es la DESVIACION A LA BAJA estandar: la raiz del
promedio de min(r, 0)^2 sobre TODOS los dias, no la desviacion estandar del
subconjunto de dias negativos. La diferencia importa: tomar la desviacion del
subconjunto resta la media de los dias malos, con lo que una serie con perdidas
grandes pero parejas aparece como poco riesgosa, y ademas divide entre el numero
de dias negativos en lugar de entre el total, lo que infla el ratio cuando hay
pocos dias malos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _anualizacion() -> float:
    """Factor de anualizacion de la desviacion estandar diaria."""
    return float(np.sqrt(config.DIAS_ANUALIZACION))


def sharpe(retornos: pd.Series) -> float:
    """Sharpe anualizado con tasa libre de riesgo cero."""
    desviacion = retornos.std(ddof=1)
    if not np.isfinite(desviacion) or desviacion == 0.0:
        return 0.0
    return float(retornos.mean() / desviacion * _anualizacion())


def desviacion_a_la_baja(retornos: pd.Series) -> float:
    """Desviacion a la baja: raiz del promedio de min(r, 0)^2 sobre TODOS los dias.

    Args:
        retornos: serie de retornos diarios.

    Returns:
        La desviacion a la baja diaria, sin anualizar.
    """
    bajadas = np.minimum(retornos.to_numpy(dtype=float), 0.0)
    return float(np.sqrt(np.mean(bajadas**2)))


def sortino(retornos: pd.Series) -> float:
    """Sortino anualizado, con la desviacion a la baja estandar."""
    desviacion = desviacion_a_la_baja(retornos)
    if not np.isfinite(desviacion) or desviacion == 0.0:
        return 0.0
    return float(retornos.mean() / desviacion * _anualizacion())


def maximo_drawdown(capital: pd.Series) -> float:
    """Drawdown maximo de una curva de capital, como numero negativo."""
    return float((capital / capital.cummax() - 1.0).min())


def cagr(capital: pd.Series, capital_inicial: float) -> float:
    """Tasa de crecimiento anual compuesta de la curva de capital."""
    anos = len(capital) / config.DIAS_ANUALIZACION
    if anos <= 0 or capital_inicial <= 0 or capital.iloc[-1] <= 0:
        return 0.0
    return float((capital.iloc[-1] / capital_inicial) ** (1.0 / anos) - 1.0)


def calmar(capital: pd.Series, capital_inicial: float) -> float:
    """Calmar: CAGR dividido por el valor absoluto del maximo drawdown."""
    drawdown = maximo_drawdown(capital)
    if drawdown == 0.0:
        return 0.0
    return float(cagr(capital, capital_inicial) / abs(drawdown))


def tasa_de_acierto(operaciones: pd.DataFrame) -> float:
    """Fraccion de operaciones cerradas con P&L neto positivo."""
    if operaciones.empty:
        return 0.0
    return float((operaciones["pnl_neto"] > 0).mean())


def profit_factor(operaciones: pd.DataFrame) -> float:
    """Cociente entre la ganancia total de ganadoras y la perdida de perdedoras."""
    if operaciones.empty:
        return 0.0
    ganancias = operaciones.loc[operaciones["pnl_neto"] > 0, "pnl_neto"].sum()
    perdidas = operaciones.loc[operaciones["pnl_neto"] < 0, "pnl_neto"].sum()
    if perdidas == 0.0:
        return float("inf") if ganancias > 0 else 0.0
    return float(ganancias / abs(perdidas))


def tarjeta_desempeno(
    serie: pd.DataFrame, operaciones: pd.DataFrame, capital_inicial: float
) -> dict:
    """Calcula las seis metricas de la tarjeta de desempeno mas contexto.

    Args:
        serie: serie diaria con columnas 'capital' y 'retorno'.
        operaciones: bitacora de operaciones con columna 'pnl_neto'.
        capital_inicial: capital al inicio del tramo.

    Returns:
        Diccionario de metricas.
    """
    capital = serie["capital"]
    retornos = serie["retorno"]
    return {
        "capital_inicial": float(capital_inicial),
        "capital_final": float(capital.iloc[-1]),
        "retorno_total": float(capital.iloc[-1] / capital_inicial - 1.0),
        "cagr": cagr(capital, capital_inicial),
        "volatilidad_anual": float(retornos.std(ddof=1) * _anualizacion()),
        "sharpe": sharpe(retornos),
        "sortino": sortino(retornos),
        "calmar": calmar(capital, capital_inicial),
        "max_drawdown": maximo_drawdown(capital),
        "win_rate": tasa_de_acierto(operaciones),
        "profit_factor": profit_factor(operaciones),
        "n_operaciones": int(len(operaciones)),
        "n_dias": int(len(serie)),
    }


def metricas_por_particion(
    serie: pd.DataFrame, operaciones: pd.DataFrame, capital_inicial: float
) -> dict:
    """Tarjeta de desempeno global y por cada particion.

    Para cada particion se usa como capital inicial el capital con el que se
    entro a ese tramo, de modo que las metricas de prueba y validacion no
    arrastren la ganancia de entrenamiento.

    Args:
        serie: serie diaria con 'capital', 'retorno' y 'particion'.
        operaciones: bitacora con 'pnl_neto' y 'particion'.
        capital_inicial: capital al inicio de la muestra.

    Returns:
        Diccionario `{'global': {...}, 'entrenamiento': {...}, ...}`.
    """
    resultado = {
        "global": tarjeta_desempeno(serie, operaciones, capital_inicial)
    }
    capital_entrada = capital_inicial
    for particion in config.NOMBRES_PARTICIONES:
        tramo = serie[serie["particion"] == particion]
        if tramo.empty:
            continue
        operaciones_tramo = (
            operaciones[operaciones["particion"] == particion]
            if "particion" in operaciones.columns
            else operaciones.iloc[0:0]
        )
        resultado[particion] = tarjeta_desempeno(
            tramo, operaciones_tramo, capital_entrada
        )
        capital_entrada = float(tramo["capital"].iloc[-1])
    return resultado


def sharpe_movil(retornos: pd.Series, ventana: int = config.DIAS_ANUALIZACION) -> pd.Series:
    """Sharpe movil anualizado sobre una ventana de `ventana` dias habiles.

    Args:
        retornos: serie de retornos diarios.
        ventana: tamano de la ventana, por defecto 252 dias.

    Returns:
        Serie del Sharpe movil, con NaN en las primeras `ventana - 1` fechas.
    """
    media = retornos.rolling(ventana).mean()
    desviacion = retornos.rolling(ventana).std(ddof=1)
    return (media / desviacion * _anualizacion()).rename(f"sharpe_movil_{ventana}")
