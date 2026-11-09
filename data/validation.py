"""Validacion de calidad del panel de precios.

Separa fallas FATALES (invalidan el panel y detienen el pipeline) de
OBSERVACIONES (se reportan para revision humana pero no detienen nada). El
criterio: una falla fatal rompe un supuesto del modelo; una observacion solo
senala un dato que conviene mirar antes de interpretar resultados.
"""

from __future__ import annotations

import pandas as pd

import config


def _maximo_nan_consecutivos(serie: pd.Series) -> int:
    """Cuenta la racha mas larga de NaN consecutivos de una serie."""
    es_nan = serie.isna()
    if not es_nan.any():
        return 0
    bloques = (~es_nan).cumsum()
    return int(es_nan.groupby(bloques).sum().max())


def validar_panel(panel: pd.DataFrame, precios_crudos: pd.DataFrame) -> dict:
    """Corre todas las validaciones y devuelve un reporte estructurado.

    Args:
        panel: panel limpio (salida de `data.cleaning.construir_panel`).
        precios_crudos: panel antes de limpiar, para contar cuanto se relleno.

    Returns:
        Diccionario con llaves 'resumen', 'por_ticker', 'fatales' y
        'observaciones'.
    """
    fatales: list[str] = []
    observaciones: list[str] = []

    if panel.isna().any().any():
        fatales.append("El panel limpio todavia contiene NaN.")
    if not panel.index.is_monotonic_increasing:
        fatales.append("El indice de fechas no es monotono creciente.")
    if panel.index.has_duplicates:
        fatales.append("El indice de fechas tiene fechas duplicadas.")

    no_positivos = int((panel <= 0).sum().sum())
    if no_positivos:
        fatales.append(f"Hay {no_positivos} precios menores o iguales a cero.")

    anos = (panel.index.max() - panel.index.min()).days / 365.25
    if not 10.0 <= anos <= 15.0:
        fatales.append(
            f"La ventana es de {anos:.2f} anos, fuera del rango 10-15 que exige "
            "la seccion 3.1 de los lineamientos."
        )

    retornos = panel.pct_change().dropna(how="all")
    extremos = retornos.abs() > config.RETORNO_DIARIO_SOSPECHOSO
    n_extremos = int(extremos.to_numpy().sum())
    if n_extremos:
        detalle = [
            f"{ticker} {fecha.date()} {retornos.at[fecha, ticker]:+.1%}"
            for ticker in panel.columns
            for fecha in retornos.index[extremos[ticker]]
        ]
        observaciones.append(
            f"{n_extremos} retornos diarios con valor absoluto mayor a "
            f"{config.RETORNO_DIARIO_SOSPECHOSO:.0%}: " + "; ".join(detalle)
        )

    por_ticker = pd.DataFrame(
        {
            "primer_dato": precios_crudos.apply(lambda s: s.first_valid_index()),
            "ultimo_dato": precios_crudos.apply(lambda s: s.last_valid_index()),
            "nan_crudos": precios_crudos.isna().sum(),
            "max_nan_consecutivos": precios_crudos.apply(_maximo_nan_consecutivos),
            "obs_limpias": panel.notna().sum(),
            "precio_min": panel.min().round(2),
            "precio_max": panel.max().round(2),
            "vol_anual": (retornos.std() * config.DIAS_ANUALIZACION**0.5).round(4),
        }
    )

    excedidos = por_ticker.index[
        por_ticker["max_nan_consecutivos"] > config.MAXIMO_NAN_CONSECUTIVOS
    ].tolist()
    if excedidos:
        fatales.append(
            f"Tickers con mas de {config.MAXIMO_NAN_CONSECUTIVOS} NaN "
            f"consecutivos en el panel crudo: {excedidos}"
        )

    rellenados = int(
        precios_crudos.loc[panel.index, panel.columns].isna().to_numpy().sum()
    )
    resumen = {
        "n_observaciones": int(len(panel)),
        "n_tickers": int(panel.shape[1]),
        "fecha_inicio": str(panel.index.min().date()),
        "fecha_fin": str(panel.index.max().date()),
        "anos_de_historia": round(anos, 2),
        "nan_rellenados": rellenados,
        "retornos_extremos": n_extremos,
    }

    return {
        "resumen": resumen,
        "por_ticker": por_ticker,
        "fatales": fatales,
        "observaciones": observaciones,
    }


def exigir_panel_valido(reporte: dict) -> None:
    """Detiene el pipeline si la validacion encontro fallas fatales.

    Args:
        reporte: salida de `validar_panel`.

    Raises:
        ValueError: si hay al menos una falla fatal.
    """
    if reporte["fatales"]:
        detalle = "\n  - ".join(reporte["fatales"])
        raise ValueError(f"Validacion de datos fallida:\n  - {detalle}")


def formatear_reporte(reporte: dict) -> str:
    """Convierte el reporte de validacion en texto legible para la consola."""
    lineas = ["Resumen del panel:"]
    lineas += [f"  {clave}: {valor}" for clave, valor in reporte["resumen"].items()]
    lineas += ["", "Diagnostico por ticker:", reporte["por_ticker"].to_string()]
    if reporte["observaciones"]:
        lineas += ["", "Observaciones (no detienen el pipeline):"]
        lineas += [f"  - {obs}" for obs in reporte["observaciones"]]
    return "\n".join(lineas)
