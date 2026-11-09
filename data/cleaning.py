"""Limpieza y alineacion del panel de precios.

Decisiones de limpieza declaradas (todas defendibles en la exposicion):

1. Se opera con el CIERRE AJUSTADO por splits y dividendos. Sin ajustar, cada
   split mete un salto de nivel que ninguna relacion de cointegracion puede
   absorber y que el filtro de Kalman interpretaria como un cambio estructural
   de los pesos.
2. Se trabaja en NIVELES de precio, no en logaritmos, para que el vector de
   pesos se traduzca directamente a numero de acciones y la contabilidad
   multi-pata cierre al centavo (prueba obligatoria 3).
3. Los NaN aislados se rellenan hacia adelante con un limite corto. Rellenar
   hacia adelante usa solo informacion pasada, asi que no introduce look-ahead;
   rellenar hacia atras si lo haria y esta prohibido.
4. La ventana se recorta al primer dia en que TODOS los tickers cotizan, para
   que todos los grupos candidatos se evaluen sobre la misma muestra.
"""

from __future__ import annotations

import pandas as pd

import config


def eliminar_dias_sin_mercado(precios: pd.DataFrame) -> pd.DataFrame:
    """Quita fechas duplicadas y filas donde ningun ticker cotizo.

    Args:
        precios: panel crudo de precios.

    Returns:
        Panel sin filas vacias ni fechas repetidas, ordenado por fecha.
    """
    limpio = precios[~precios.index.duplicated(keep="first")].sort_index()
    return limpio.dropna(how="all")


def recortar_ventana_comun(precios: pd.DataFrame) -> pd.DataFrame:
    """Recorta el panel al tramo en que todos los tickers tienen cotizacion.

    El inicio es la fecha mas tardia entre las primeras cotizaciones de cada
    ticker; el fin es la mas temprana entre las ultimas.

    Args:
        precios: panel de precios con posibles NaN al inicio por tickers jovenes.

    Returns:
        Panel recortado a la ventana comun.

    Raises:
        ValueError: si la ventana comun queda vacia.
    """
    primeras = precios.apply(lambda serie: serie.first_valid_index())
    ultimas = precios.apply(lambda serie: serie.last_valid_index())
    inicio, fin = primeras.max(), ultimas.min()
    if pd.isna(inicio) or pd.isna(fin) or inicio >= fin:
        raise ValueError("La ventana comun del universo esta vacia.")
    return precios.loc[inicio:fin].copy()


def rellenar_huecos_aislados(
    precios: pd.DataFrame, limite: int = config.MAXIMO_NAN_CONSECUTIVOS
) -> pd.DataFrame:
    """Rellena hacia adelante huecos cortos y descarta las filas restantes.

    Un hueco corto es un dia en que un ticker no reporto cierre aunque el
    mercado abrio (suspension intradia, error de la fuente). Un hueco largo
    indica un problema real de datos y su fila se descarta.

    Args:
        precios: panel ya recortado a la ventana comun.
        limite: maximo de dias consecutivos que se permite rellenar.

    Returns:
        Panel sin NaN.
    """
    rellenado = precios.ffill(limit=limite)
    return rellenado.dropna(how="any")


def construir_panel(precios_crudos: pd.DataFrame) -> pd.DataFrame:
    """Aplica la secuencia completa de limpieza al panel crudo.

    Args:
        precios_crudos: salida de `data.loader.cargar_precios`.

    Returns:
        Panel de precios limpio, sin NaN, en la ventana comun del universo.
    """
    panel = eliminar_dias_sin_mercado(precios_crudos)
    panel = recortar_ventana_comun(panel)
    panel = rellenar_huecos_aislados(panel)
    panel.index.name = "fecha"
    panel.columns.name = None
    return panel
