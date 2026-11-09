"""Grid search de umbrales del z-score sobre entrenamiento (seccion 3.6).

Procedimiento declarado, que es lo que el PDF pide documentar:

1. La malla es el producto cartesiano de `MALLA_Z_ENTRADA`, `MALLA_Z_SALIDA` y
   `MALLA_Z_STOP`, descartando combinaciones incoherentes (salida por encima de
   entrada, o stop por debajo de entrada). El periodo maximo de tenencia NO se
   busca: se calibra a la vida media del spread, segun la seccion 3.6.
2. Cada combinacion se evalua con el MISMO motor en dolares que produce el
   resultado final (`backtest.engine.simular_grupo_en_dolares`, envoltorio de
   `walkforward.simular_consolidado`), restringido a la particion de
   ENTRENAMIENTO. Esto importa: la version anterior calibraba sobre una
   contabilidad en "unidades de nocional bruto" que ignoraba los precios y, por
   tanto, describia un portafolio distinto del que despues se operaba.
3. La funcion objetivo es el Sharpe anualizado del retorno NETO de costos. Se
   usa el neto, no el bruto, porque una malla optimizada sobre el bruto elegiria
   siempre el umbral de entrada mas bajo (mas operaciones, mas senal) y la
   comision de 0.125% por pata se las comeria todas fuera de muestra.
4. Se exige un minimo de operaciones para que el Sharpe sea interpretable; una
   combinacion con tres operaciones en ocho anos puede tener Sharpe altisimo por
   azar. Las combinaciones que no lo cumplen se reportan pero no se eligen.
5. Desempate: a igual Sharpe, mayor numero de operaciones, que es la
   combinacion con mas evidencia detras.

Nada de esto mira prueba ni validacion.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import config
from backtest import engine
from strategy import signals

#: Minimo de operaciones en entrenamiento para considerar una combinacion.
MINIMO_OPERACIONES = 10


def generar_malla(tenencia_maxima: int) -> list[signals.Umbrales]:
    """Construye la malla de umbrales coherentes.

    Args:
        tenencia_maxima: dias habiles maximos, ya calibrados a la vida media.

    Returns:
        Lista de combinaciones de umbrales a evaluar.
    """
    combinaciones = []
    for z_entrada, z_salida, z_stop in itertools.product(
        config.MALLA_Z_ENTRADA, config.MALLA_Z_SALIDA, config.MALLA_Z_STOP
    ):
        if z_salida >= z_entrada or z_stop <= z_entrada:
            continue
        combinaciones.append(
            signals.Umbrales(
                z_entrada=z_entrada,
                z_salida=z_salida,
                z_stop=z_stop,
                tenencia_maxima=tenencia_maxima,
            )
        )
    return combinaciones


def contar_operaciones(posicion: pd.Series) -> int:
    """Cuenta los tramos con posicion abierta en una serie de direcciones."""
    valores = posicion.to_numpy()
    previos = np.concatenate([[0], valores[:-1]])
    return int(np.sum((valores != 0) & (valores != previos)))


def buscar_umbrales(
    precios_entrenamiento: pd.DataFrame,
    betas_entrenamiento: pd.DataFrame,
    z_entrenamiento: pd.Series,
    operable_entrenamiento: pd.Series,
    vida_media_dias: float,
    grupo: str,
) -> tuple[signals.Umbrales, pd.DataFrame]:
    """Corre el grid search en entrenamiento y devuelve los umbrales elegidos.

    Args:
        precios_entrenamiento: panel del grupo en entrenamiento.
        betas_entrenamiento: betas del filtro 1 en entrenamiento.
        z_entrenamiento: z-score en entrenamiento.
        operable_entrenamiento: bandera de burn-in en entrenamiento.
        vida_media_dias: vida media del spread estimada en entrenamiento.
        grupo: etiqueta del grupo.

    Returns:
        Tupla `(umbrales_elegidos, tabla)` con la malla completa evaluada.

    Raises:
        ValueError: si la malla queda vacia.
    """
    tenencia = signals.tenencia_maxima_desde_vida_media(vida_media_dias)
    malla = generar_malla(tenencia)
    if not malla:
        raise ValueError("La malla de umbrales quedo vacia.")

    filas = []
    for umbrales in malla:
        tabla_senales = signals.generar_senales(
            z_entrenamiento, operable_entrenamiento, umbrales
        )
        posicion = tabla_senales["posicion"]
        resultado = engine.simular_grupo_en_dolares(
            precios_entrenamiento, betas_entrenamiento, posicion, grupo
        )
        serie = resultado["serie"]
        n_operaciones = contar_operaciones(posicion)
        filas.append(
            {
                **umbrales.como_dict(),
                "sharpe_neto": engine.sharpe_anualizado(serie["retorno"]),
                "capital_final": float(serie["capital"].iloc[-1]),
                "retorno_total": float(
                    serie["capital"].iloc[-1] / config.CAPITAL_INICIAL - 1.0
                ),
                "pnl_bruto": float(serie["pnl_bruto"].sum()),
                "comision_total": float(serie["comision"].sum()),
                "prestamo_total": float(serie["prestamo"].sum()),
                "n_operaciones": n_operaciones,
                "dias_en_mercado": int((posicion != 0).sum()),
            }
        )

    tabla = pd.DataFrame(filas)
    elegibles = tabla[tabla["n_operaciones"] >= MINIMO_OPERACIONES]
    if elegibles.empty:
        elegibles = tabla

    orden = elegibles.sort_values(
        ["sharpe_neto", "n_operaciones"], ascending=[False, False]
    )
    mejor = orden.iloc[0]
    tabla["elegido"] = (
        (tabla["z_entrada"] == mejor["z_entrada"])
        & (tabla["z_salida"] == mejor["z_salida"])
        & (tabla["z_stop"] == mejor["z_stop"])
    )
    tabla["elegible"] = tabla["n_operaciones"] >= MINIMO_OPERACIONES

    umbrales_elegidos = signals.Umbrales(
        z_entrada=float(mejor["z_entrada"]),
        z_salida=float(mejor["z_salida"]),
        z_stop=float(mejor["z_stop"]),
        tenencia_maxima=tenencia,
    )
    return umbrales_elegidos, tabla.sort_values("sharpe_neto", ascending=False)
