"""Politica de rebalanceo: diario contra banda de no-rebalanceo (seccion 3.6).

La seccion 3.6 pide aplicar rebalanceo diario de los pesos conforme evoluciona
beta_t y "cuantificar su costo comparandolo contra una politica de banda de
no-rebalanceo". Aqui viven la banda y la calibracion de su ancho.

La banda opera sobre ACCIONES
-----------------------------
Es lo que se negocia y lo que genera comision. Con q* las acciones objetivo del
dia y q las que se traen, se rebalancea si la desviacion mas grande de una pata,
medida en dolares y relativa al nocional bruto objetivo, supera el ancho:

    max_j |q*_j - q_j| P_j / sum_k |q*_k| P_k  >  ancho

Dos casos fuerzan rebalanceo sin importar la banda, porque no son un ajuste de
pesos sino un cambio de posicion: abrir (de plano a invertido) y cerrar (de
invertido a plano). Si no se forzaran, una senal de salida podria quedar sin
ejecutar y la posicion seguiria abierta contra la regla de trading.

Calibracion del ancho
---------------------
El ancho se elige por grid search sobre `config.MALLA_BANDA_NO_REBALANCEO`
maximizando el Sharpe neto SOLO en entrenamiento, con el MISMO motor en dolares
con que se mide el resultado final. No se mira prueba ni validacion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def aplicar_banda_a_acciones(
    objetivo: np.ndarray, sostenidas: np.ndarray, precios: np.ndarray, ancho: float
) -> np.ndarray:
    """Decide las acciones efectivas del dia bajo la banda de no-rebalanceo.

    Args:
        objetivo: acciones objetivo del dia.
        sostenidas: acciones que se traen del dia anterior.
        precios: precios del dia, en el mismo orden.
        ancho: ancho de la banda como fraccion del nocional bruto objetivo.

    Returns:
        Las acciones objetivo si toca rebalancear, o las sostenidas si no.
    """
    abriendo = np.allclose(sostenidas, 0.0) and not np.allclose(objetivo, 0.0)
    cerrando = np.allclose(objetivo, 0.0) and not np.allclose(sostenidas, 0.0)
    if abriendo or cerrando:
        return objetivo

    bruto_objetivo = float(np.abs(objetivo) @ precios)
    if bruto_objetivo <= 0.0:
        return objetivo
    desvio = float(np.max(np.abs(objetivo - sostenidas) * precios) / bruto_objetivo)
    return objetivo if desvio > ancho else sostenidas


def calibrar_ancho_banda(
    precios_entrenamiento: pd.DataFrame,
    betas_entrenamiento: pd.DataFrame,
    posicion_entrenamiento: pd.Series,
    grupo: str,
    malla: tuple[float, ...] = config.MALLA_BANDA_NO_REBALANCEO,
) -> tuple[float, pd.DataFrame]:
    """Elige el ancho de banda por Sharpe neto, usando SOLO entrenamiento.

    La politica que opera la estrategia es el rebalanceo DIARIO, porque es lo
    que pide la seccion 3.6. El ancho se calibra para que la comparacion sea
    justa: se le da a la banda su mejor configuracion en entrenamiento, con el
    mismo criterio y el mismo motor con que se eligieron los umbrales.

    Args:
        precios_entrenamiento: panel del grupo en entrenamiento.
        betas_entrenamiento: betas del filtro 1 en entrenamiento.
        posicion_entrenamiento: direccion por dia en entrenamiento.
        grupo: etiqueta del grupo.
        malla: anchos candidatos.

    Returns:
        Tupla `(ancho_elegido, tabla)`.
    """
    from backtest import engine

    filas = []
    for ancho in malla:
        resultado = engine.simular_grupo_en_dolares(
            precios_entrenamiento,
            betas_entrenamiento,
            posicion_entrenamiento,
            grupo,
            ancho_banda=ancho,
        )
        serie = resultado["serie"]
        filas.append(
            {
                "ancho_banda": ancho,
                "sharpe_neto": engine.sharpe_anualizado(serie["retorno"]),
                "capital_final": float(serie["capital"].iloc[-1]),
                "comision_total": float(serie["comision"].sum()),
                "prestamo_total": float(serie["prestamo"].sum()),
                "dias_con_operacion": int((serie["turnover_usd"] > 1e-6).sum()),
                "turnover_usd": float(serie["turnover_usd"].sum()),
            }
        )

    tabla = pd.DataFrame(filas)
    ancho_elegido = float(tabla.loc[tabla["sharpe_neto"].idxmax(), "ancho_banda"])
    tabla["elegido"] = tabla["ancho_banda"] == ancho_elegido
    return ancho_elegido, tabla


def comparar_politicas(
    precios: pd.DataFrame,
    betas: pd.DataFrame,
    posicion: pd.Series,
    grupo: str,
    ancho: float,
    etiquetas: pd.Series,
) -> pd.DataFrame:
    """Compara costo y desempeno de las dos politicas por particion.

    Ambas se simulan con el mismo motor en dolares sobre la muestra completa,
    con el grupo operado por separado, de modo que la comparacion aisle el
    efecto de la politica de rebalanceo.

    Args:
        precios: panel del grupo.
        betas: betas del filtro 1.
        posicion: direccion por dia.
        grupo: etiqueta del grupo.
        ancho: ancho de banda calibrado en entrenamiento.
        etiquetas: serie de particiones por fecha.

    Returns:
        DataFrame con una fila por politica y particion.
    """
    from backtest import engine

    simulaciones = {
        "diario": engine.simular_grupo_en_dolares(precios, betas, posicion, grupo),
        "banda": engine.simular_grupo_en_dolares(
            precios, betas, posicion, grupo, ancho_banda=ancho
        ),
    }

    filas = []
    for nombre, resultado in simulaciones.items():
        serie = resultado["serie"]
        for particion in config.NOMBRES_PARTICIONES:
            tramo = serie.loc[etiquetas.values == particion]
            filas.append(
                {
                    "grupo": grupo,
                    "politica": nombre,
                    "ancho_banda": ancho if nombre == "banda" else 0.0,
                    "particion": particion,
                    "turnover_usd": float(tramo["turnover_usd"].sum()),
                    "comision_total": float(tramo["comision"].sum()),
                    "prestamo_total": float(tramo["prestamo"].sum()),
                    "pnl_bruto": float(tramo["pnl_bruto"].sum()),
                    "pnl_neto": float(tramo["pnl_neto"].sum()),
                    "sharpe_neto": engine.sharpe_anualizado(tramo["retorno"]),
                    "dias_con_operacion": int((tramo["turnover_usd"] > 1e-6).sum()),
                }
            )
    return pd.DataFrame(filas)
