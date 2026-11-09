"""Particion cronologica 60/20/20 del panel (seccion 3.1).

La particion es la primera linea de defensa contra el look-ahead: TODO lo que
se estima (screening de correlacion, Engle-Granger, Johansen, calibracion de Q
y R, grid search de umbrales y pesos de Risk Parity) se calcula exclusivamente
con `entrenamiento`. `prueba` y `validacion` solo se recorren hacia adelante
aplicando parametros ya congelados.

Convencion de redondeo declarada: se asignan `floor(0.60 * N)` observaciones a
entrenamiento y `floor(0.20 * N)` a prueba, y el residuo a validacion. Asi
ninguna particion queda vacia y la suma es exactamente N; validacion absorbe
el redondeo porque es la particion que nunca se usa para estimar.
"""

from __future__ import annotations

import pandas as pd

import config


def calcular_indices_de_corte(n_observaciones: int) -> tuple[int, int]:
    """Devuelve las posiciones que separan entrenamiento, prueba y validacion.

    Args:
        n_observaciones: numero total de fechas del panel.

    Returns:
        Tupla `(fin_entrenamiento, fin_prueba)` con posiciones enteras tipo
        slice: entrenamiento es `[0, fin_entrenamiento)`, prueba es
        `[fin_entrenamiento, fin_prueba)` y validacion es `[fin_prueba, N)`.

    Raises:
        ValueError: si alguna particion quedaria vacia.
    """
    fin_entrenamiento = int(n_observaciones * config.FRACCION_ENTRENAMIENTO)
    fin_prueba = fin_entrenamiento + int(n_observaciones * config.FRACCION_PRUEBA)
    if fin_entrenamiento <= 0 or fin_prueba <= fin_entrenamiento or fin_prueba >= n_observaciones:
        raise ValueError(
            f"La particion 60/20/20 de {n_observaciones} observaciones deja "
            "alguna particion vacia."
        )
    return fin_entrenamiento, fin_prueba


def dividir_cronologico(indice: pd.DatetimeIndex) -> dict[str, pd.DatetimeIndex]:
    """Parte un indice de fechas en las tres particiones cronologicas.

    Args:
        indice: indice de fechas ordenado del panel limpio.

    Returns:
        Diccionario `{'entrenamiento': ..., 'prueba': ..., 'validacion': ...}`
        con los sub-indices disjuntos y contiguos.
    """
    fin_entrenamiento, fin_prueba = calcular_indices_de_corte(len(indice))
    return {
        "entrenamiento": indice[:fin_entrenamiento],
        "prueba": indice[fin_entrenamiento:fin_prueba],
        "validacion": indice[fin_prueba:],
    }


def etiquetar_particiones(indice: pd.DatetimeIndex) -> pd.Series:
    """Devuelve una serie que etiqueta cada fecha con su particion.

    Esta serie viaja con todos los artifacts de series de tiempo para que el
    dashboard pueda sombrear los tramos sin recalcular los cortes.

    Args:
        indice: indice de fechas del panel limpio.

    Returns:
        Serie de strings indexada por fecha, con nombre 'particion'.
    """
    particiones = dividir_cronologico(indice)
    etiquetas = pd.Series(index=indice, dtype="object", name="particion")
    for nombre, sub_indice in particiones.items():
        etiquetas.loc[sub_indice] = nombre
    return etiquetas


def resumen_particiones(indice: pd.DatetimeIndex) -> pd.DataFrame:
    """Tabla con fechas y conteos de cada particion, para reporte y consola.

    Args:
        indice: indice de fechas del panel limpio.

    Returns:
        DataFrame con columnas 'inicio', 'fin', 'n_dias' y 'porcentaje'.
    """
    particiones = dividir_cronologico(indice)
    total = len(indice)
    filas = {
        nombre: {
            "inicio": sub.min().date(),
            "fin": sub.max().date(),
            "n_dias": len(sub),
            "porcentaje": round(100 * len(sub) / total, 2),
        }
        for nombre, sub in particiones.items()
    }
    return pd.DataFrame(filas).T.loc[list(config.NOMBRES_PARTICIONES)]


def recortar_a_particion(datos: pd.DataFrame, particion: str) -> pd.DataFrame:
    """Devuelve el subconjunto de `datos` perteneciente a una particion.

    Args:
        datos: DataFrame indexado por fecha.
        particion: 'entrenamiento', 'prueba' o 'validacion'.

    Returns:
        Sub-DataFrame con las fechas de esa particion.

    Raises:
        KeyError: si el nombre de particion no existe.
    """
    if particion not in config.NOMBRES_PARTICIONES:
        raise KeyError(
            f"Particion desconocida: {particion!r}. "
            f"Opciones: {config.NOMBRES_PARTICIONES}"
        )
    return datos.loc[dividir_cronologico(datos.index)[particion]]
