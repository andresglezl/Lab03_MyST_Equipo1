"""Capa unica de lectura y escritura de `artifacts/`.

Todo el pipeline escribe aqui y el dashboard SOLO lee de aqui (seccion 3.8:
"El dashboard debe cargar los datos desde los archivos de salida del backtest.
No debe volver a ejecutar el backtest"). Centralizar el acceso en un modulo
evita que cada fase invente su propia ruta o su propio formato.

Convencion de formatos:
  - Series de tiempo -> Parquet (tipado, compacto, conserva el indice de fechas).
  - Tablas de reporte -> CSV (se abren en Excel para la exposicion).
  - Escalares y metadatos -> JSON.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import config

NOMBRE_MANIFIESTO = "manifest.json"


def ruta(*partes: str) -> Path:
    """Construye una ruta dentro de `artifacts/` creando los directorios.

    Args:
        *partes: componentes de la ruta relativa, p. ej. ('kalman', 'f1.parquet').

    Returns:
        Ruta absoluta al archivo destino.
    """
    destino = config.DIR_ARTIFACTS.joinpath(*partes)
    destino.parent.mkdir(parents=True, exist_ok=True)
    return destino


def escribir_serie(datos: pd.DataFrame, *partes: str) -> Path:
    """Escribe una serie de tiempo indexada por fecha en Parquet."""
    destino = ruta(*partes)
    datos.to_parquet(destino)
    return destino


def leer_serie(*partes: str) -> pd.DataFrame:
    """Lee una serie de tiempo escrita por `escribir_serie`."""
    return pd.read_parquet(ruta(*partes))


def escribir_tabla(datos: pd.DataFrame, *partes: str, indice: bool = True) -> Path:
    """Escribe una tabla de reporte en CSV."""
    destino = ruta(*partes)
    datos.to_csv(destino, index=indice)
    return destino


def leer_tabla(*partes: str) -> pd.DataFrame:
    """Lee una tabla escrita por `escribir_tabla`."""
    return pd.read_csv(ruta(*partes))


def escribir_json(contenido: dict, *partes: str) -> Path:
    """Escribe un diccionario en JSON legible."""
    destino = ruta(*partes)
    with destino.open("w", encoding="utf-8") as archivo:
        json.dump(contenido, archivo, indent=2, ensure_ascii=False, default=str)
    return destino


def leer_json(*partes: str) -> dict:
    """Lee un JSON escrito por `escribir_json`."""
    with ruta(*partes).open(encoding="utf-8") as archivo:
        return json.load(archivo)


def _versiones_de_paquetes() -> dict[str, str]:
    """Versiones instaladas de las dependencias que afectan resultados."""
    import numpy
    import pandas
    import scipy
    import statsmodels
    import yfinance

    return {
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "yfinance": yfinance.__version__,
    }


def inicializar_manifiesto() -> dict:
    """Crea `artifacts/manifest.json` con los metadatos de la corrida.

    El manifiesto es el registro de reproducibilidad: deja asentada la semilla,
    las versiones de paquetes y los parametros con los que se produjo cada
    conjunto de artifacts. El dashboard lo lee para mostrar de que corrida
    provienen las cifras que esta graficando.

    Returns:
        El manifiesto recien escrito.
    """
    import sys

    manifiesto = {
        "corrida": {
            "id": datetime.now().strftime("%Y%m%dT%H%M%S"),
            "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "paquetes": _versiones_de_paquetes(),
        },
        "parametros": {
            "semilla": config.SEMILLA,
            "fecha_inicio": config.FECHA_INICIO,
            "fecha_fin": config.FECHA_FIN,
            "division": {
                "entrenamiento": config.FRACCION_ENTRENAMIENTO,
                "prueba": config.FRACCION_PRUEBA,
                "validacion": config.FRACCION_VALIDACION,
            },
            "capital_inicial": config.CAPITAL_INICIAL,
            "comision_por_pata": config.COMISION_POR_PATA,
            "tasa_prestamo_anual": config.TASA_PRESTAMO_ANUAL,
            "dias_devengo_prestamo": config.DIAS_DEVENGO_PRESTAMO,
            "dias_anualizacion": config.DIAS_ANUALIZACION,
            "exposicion_maxima": config.EXPOSICION_MAXIMA,
            "correlacion_minima": config.CORRELACION_MINIMA,
            "pvalue_maximo": config.PVALUE_MAXIMO,
            "burn_in_dias": config.BURN_IN_DIAS,
        },
        "etapas": {},
    }
    escribir_json(manifiesto, NOMBRE_MANIFIESTO)
    return manifiesto


def actualizar_manifiesto(etapa: str, contenido: dict) -> dict:
    """Agrega o reemplaza la entrada de una etapa en el manifiesto.

    Args:
        etapa: nombre de la etapa del pipeline, p. ej. 'datos'.
        contenido: metadatos y lista de salidas producidas por esa etapa.

    Returns:
        El manifiesto actualizado.
    """
    manifiesto = (
        leer_json(NOMBRE_MANIFIESTO)
        if ruta(NOMBRE_MANIFIESTO).exists()
        else inicializar_manifiesto()
    )
    manifiesto.setdefault("etapas", {})[etapa] = contenido
    escribir_json(manifiesto, NOMBRE_MANIFIESTO)
    return manifiesto
