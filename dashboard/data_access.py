"""Acceso de SOLO LECTURA a `artifacts/` para el dashboard (seccion 3.8).

El requisito es explicito: "El dashboard debe cargar los datos desde los
archivos de salida del backtest. No debe volver a ejecutar el backtest". Este
modulo es la unica puerta por la que el dashboard toca datos, y no importa
ningun modulo de `backtest/`, `kalman/`, `strategy/` ni `cointegration/`: solo
`artifacts_io`. Si alguien intentara recalcular algo desde el dashboard, tendria
que agregar un import aqui y seria evidente en la revision.

Todas las funciones usan `st.cache_data` para que cambiar un filtro en la
interfaz no vuelva a leer el disco.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import artifacts_io


@st.cache_data(show_spinner=False)
def manifiesto() -> dict:
    """Metadatos de la corrida: semilla, versiones, parametros y etapas."""
    return artifacts_io.leer_json("manifest.json")


@st.cache_data(show_spinner=False)
def precios() -> pd.DataFrame:
    """Panel de precios con la columna de particion."""
    return artifacts_io.leer_serie("prices.parquet")


@st.cache_data(show_spinner=False)
def grupos_probados() -> pd.DataFrame:
    """Tabla completa de los grupos evaluados por el embudo de seleccion."""
    return artifacts_io.leer_tabla("selection", "groups_tested.csv")


@st.cache_data(show_spinner=False)
def filtro1(grupo: str) -> pd.DataFrame:
    """Serie del filtro de Kalman 1: betas, ganancias y varianzas."""
    return artifacts_io.leer_serie("kalman", f"filter1_{grupo}.parquet")


@st.cache_data(show_spinner=False)
def filtro2(grupo: str) -> pd.DataFrame:
    """Serie del filtro de Kalman 2: media y varianza latentes, z-score."""
    return artifacts_io.leer_serie("kalman", f"filter2_{grupo}.parquet")


@st.cache_data(show_spinner=False)
def senales(grupo: str) -> pd.DataFrame:
    """Serie diaria de senales, posiciones y costos del grupo."""
    return artifacts_io.leer_serie("strategy", f"signals_{grupo}.parquet")


@st.cache_data(show_spinner=False)
def johansen_movil(grupo: str) -> pd.DataFrame:
    """Primer eigenvector de Johansen reestimado en ventanas moviles."""
    return artifacts_io.leer_serie("rolling", f"johansen_eigen_{grupo}.parquet")


@st.cache_data(show_spinner=False)
def capital_consolidado() -> pd.DataFrame:
    """Curva de capital del portafolio consolidado con Risk Parity."""
    return artifacts_io.leer_serie("backtest", "portfolio_equity.parquet")


@st.cache_data(show_spinner=False)
def capital_pesos_iguales() -> pd.DataFrame:
    """Curva de capital del portafolio con pesos iguales entre grupos."""
    return artifacts_io.leer_serie("backtest", "equity_equal_weight.parquet")


@st.cache_data(show_spinner=False)
def capital_grupo(grupo: str) -> pd.DataFrame:
    """Curva de capital del grupo operado por separado."""
    return artifacts_io.leer_serie("backtest", f"group_equity_{grupo}.parquet")


@st.cache_data(show_spinner=False)
def operaciones() -> pd.DataFrame:
    """Bitacora de operaciones de todos los grupos, con P&L en dolares."""
    tabla = artifacts_io.leer_tabla("backtest", "trades_all.csv")
    for columna in ("fecha_entrada", "fecha_salida"):
        tabla[columna] = pd.to_datetime(tabla[columna])
    return tabla


@st.cache_data(show_spinner=False)
def metricas() -> dict:
    """Tarjetas de desempeno por portafolio y particion."""
    return artifacts_io.leer_json("performance", "metrics.json")


@st.cache_data(show_spinner=False)
def atribucion_costos() -> pd.DataFrame:
    """Descomposicion de P&L bruto en neto, comisiones y prestamo."""
    return artifacts_io.leer_tabla("performance", "cost_attribution.csv")


@st.cache_data(show_spinner=False)
def sharpe_movil() -> pd.DataFrame:
    """Sharpe movil de 252 dias de todos los portafolios."""
    return artifacts_io.leer_serie("performance", "rolling_sharpe.parquet")


@st.cache_data(show_spinner=False)
def pesos_risk_parity() -> pd.DataFrame:
    """Pesos y contribuciones al riesgo de Risk Parity y de pesos iguales."""
    return artifacts_io.leer_tabla("riskparity", "weights.csv")


@st.cache_data(show_spinner=False)
def correlacion_spreads() -> pd.DataFrame:
    """Matriz de correlacion entre los spreads de los grupos."""
    return pd.read_csv(
        artifacts_io.ruta("riskparity", "spread_corr.csv"), index_col=0
    )


@st.cache_data(show_spinner=False)
def atribucion_drawdown() -> pd.DataFrame:
    """Atribucion del drawdown maximo del consolidado a los grupos."""
    return artifacts_io.leer_tabla("performance", "drawdown_attribution.csv")


@st.cache_data(show_spinner=False)
def diagnostico_filtros() -> pd.DataFrame:
    """Varianza y autocorrelacion de las innovaciones de ambos filtros."""
    return artifacts_io.leer_tabla("performance", "filter_diagnostics.csv")


@st.cache_data(show_spinner=False)
def comparacion_rebalanceo() -> pd.DataFrame:
    """Rebalanceo diario contra banda de no-rebalanceo, por particion."""
    return artifacts_io.leer_tabla("backtest", "rebalance_comparison.csv")


def grupos_operados() -> list[dict]:
    """Metadatos de los grupos operados, leidos del manifiesto."""
    return manifiesto()["etapas"]["estrategia"]["grupos"]


def hay_artifacts() -> bool:
    """Indica si el pipeline ya produjo artifacts que el dashboard pueda leer."""
    return artifacts_io.ruta("manifest.json").exists()
