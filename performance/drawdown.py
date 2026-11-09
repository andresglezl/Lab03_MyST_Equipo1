"""Drawdown y su atribucion por grupo (seccion 3.7).

La seccion 3.7 pide "atribuir el drawdown maximo del portafolio consolidado a
los grupos que lo originaron". El procedimiento es:

1. Localizar la ventana del drawdown maximo: el dia del pico anterior y el dia
   del valle.
2. Sumar, dentro de esa ventana, el P&L neto diario en dolares de cada grupo.
3. Expresar cada suma como fraccion de la caida total de capital en la ventana.

La atribucion es exacta y suma 100% porque el capital del consolidado es
aditivo: cada dia, el cambio de capital es la suma de los P&L netos de los
grupos. No hay termino de interaccion que repartir.
"""

from __future__ import annotations

import pandas as pd


def serie_drawdown(capital: pd.Series) -> pd.DataFrame:
    """Serie de maximo historico y drawdown de una curva de capital.

    Args:
        capital: serie de capital indexada por fecha.

    Returns:
        DataFrame con 'maximo_historico' y 'drawdown'.
    """
    maximo = capital.cummax()
    return pd.DataFrame({"maximo_historico": maximo, "drawdown": capital / maximo - 1.0})


def ventana_maximo_drawdown(capital: pd.Series) -> dict:
    """Identifica el pico, el valle y la magnitud del drawdown maximo.

    Args:
        capital: serie de capital indexada por fecha.

    Returns:
        Diccionario con 'fecha_pico', 'fecha_valle', 'capital_pico',
        'capital_valle', 'drawdown' y 'dias_habiles'.
    """
    drawdown = serie_drawdown(capital)["drawdown"]
    fecha_valle = drawdown.idxmin()
    fecha_pico = capital.loc[:fecha_valle].idxmax()
    return {
        "fecha_pico": fecha_pico,
        "fecha_valle": fecha_valle,
        "capital_pico": float(capital.loc[fecha_pico]),
        "capital_valle": float(capital.loc[fecha_valle]),
        "drawdown": float(drawdown.loc[fecha_valle]),
        "dias_habiles": int(capital.loc[fecha_pico:fecha_valle].shape[0] - 1),
    }


def atribuir_drawdown(capital: pd.Series, pnl_por_grupo: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Atribuye el drawdown maximo del consolidado a los grupos.

    Args:
        capital: curva de capital del portafolio consolidado.
        pnl_por_grupo: P&L neto diario en dolares por grupo.

    Returns:
        Tupla `(ventana, tabla)` donde `tabla` tiene una fila por grupo con su
        P&L dentro de la ventana y su fraccion de la caida total.
    """
    ventana = ventana_maximo_drawdown(capital)
    # Se excluye el dia del pico: su P&L es el que llevo AL pico, no el que
    # empezo la caida. La ventana de perdida va del dia siguiente al valle.
    inicio = capital.index[capital.index.get_loc(ventana["fecha_pico"]) + 1]
    tramo = pnl_por_grupo.loc[inicio : ventana["fecha_valle"]]

    caida_total = ventana["capital_valle"] - ventana["capital_pico"]
    pnl_grupos = tramo.sum()

    tabla = pd.DataFrame(
        {
            "grupo": pnl_grupos.index,
            "pnl_neto_en_ventana": pnl_grupos.to_numpy().round(2),
            "fraccion_de_la_caida": (pnl_grupos / caida_total).to_numpy().round(6),
            "dias_en_ventana": len(tramo),
        }
    )
    tabla["fecha_pico"] = ventana["fecha_pico"]
    tabla["fecha_valle"] = ventana["fecha_valle"]
    tabla["caida_total"] = round(caida_total, 2)
    tabla["drawdown"] = round(ventana["drawdown"], 6)
    return ventana, tabla
