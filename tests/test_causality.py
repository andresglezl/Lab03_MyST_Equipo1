"""Prueba obligatoria 1: causalidad (seccion 3.11.1).

"El z-score en t no cambia al recalcular el pipeline completo sobre los datos
hasta t, para al menos tres valores de t."

Que demuestra: que ningun dato posterior a t influye en z_t. Si hubiera
look-ahead en cualquier parte de la cadena (normalizar con la media posterior
del filtro 2, estandarizar con la desviacion de toda la muestra, reestimar
Johansen con datos futuros), truncar la muestra en t cambiaria z_t y la prueba
fallaria.

Las tres fechas se eligen en prueba y validacion, es decir fuera de
entrenamiento, para que la calibracion de Q, R y el estado inicial sea identica
entre la corrida completa y la truncada. Asi lo unico que puede explicar una
diferencia es el look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import recalcular_z

#: Tolerancia: el z-score debe reproducirse a precision de punto flotante.
TOLERANCIA = 1e-10


def _beta_johansen(grupo: dict, grupos_cointegracion: dict) -> np.ndarray:
    """Vector cointegrante del grupo, en el orden de sus tickers."""
    beta = grupos_cointegracion[grupo["grupo"]]["beta"]
    return np.array([beta[ticker] for ticker in grupo["tickers"]])


def _fechas_de_prueba(etiquetas: pd.Series) -> list[pd.Timestamp]:
    """Tres fechas fuera de entrenamiento: inicio de prueba, media y final."""
    fuera = etiquetas.index[etiquetas.values != "entrenamiento"]
    return [fuera[20], fuera[len(fuera) // 2], fuera[-1]]


def test_z_score_no_cambia_al_truncar(
    panel, etiquetas, grupos, fin_entrenamiento, grupos_cointegracion
):
    """z_t es identico al recalcular el pipeline con datos solo hasta t."""
    fechas = _fechas_de_prueba(etiquetas)
    assert len(fechas) >= 3, "Se requieren al menos tres fechas de evaluacion."

    for grupo in grupos:
        tickers = grupo["tickers"]
        beta = _beta_johansen(grupo, grupos_cointegracion)
        completo = recalcular_z(
            panel,
            tickers,
            grupo["pata_numeraria"],
            grupo["vecm"]["vida_media_dias"],
            fin_entrenamiento,
            beta,
        )
        for fecha in fechas:
            truncado = recalcular_z(
                panel.loc[:fecha],
                tickers,
                grupo["pata_numeraria"],
                grupo["vecm"]["vida_media_dias"],
                fin_entrenamiento,
                beta,
            )
            diferencia = abs(float(truncado.loc[fecha]) - float(completo.loc[fecha]))
            assert diferencia < TOLERANCIA, (
                f"{grupo['grupo']} en {fecha.date()}: z truncado "
                f"{truncado.loc[fecha]:.12f} contra z completo "
                f"{completo.loc[fecha]:.12f}, diferencia {diferencia:.2e}"
            )


def test_z_score_del_artifact_coincide_con_el_recalculo(
    panel, grupos, fin_entrenamiento, grupos_cointegracion
):
    """El z-score guardado en artifacts/ es el mismo que produce la cadena causal.

    Cierra el circulo: no solo la cadena es causal, sino que lo que el dashboard
    lee es exactamente lo que esa cadena produce.
    """
    import artifacts_io

    for grupo in grupos:
        guardado = artifacts_io.leer_serie(
            "strategy", f"signals_{grupo['grupo']}.parquet"
        )["z"]
        recalculado = recalcular_z(
            panel,
            grupo["tickers"],
            grupo["pata_numeraria"],
            grupo["vecm"]["vida_media_dias"],
            fin_entrenamiento,
            _beta_johansen(grupo, grupos_cointegracion),
        )
        np.testing.assert_allclose(
            guardado.to_numpy(), recalculado.to_numpy(), rtol=0.0, atol=TOLERANCIA
        )


def test_particiones_son_cronologicas_y_disjuntas(etiquetas):
    """Las tres particiones son contiguas, disjuntas y en orden temporal."""
    import config
    from data import splits

    particiones = splits.dividir_cronologico(etiquetas.index)
    fin_anterior = None
    for nombre in config.NOMBRES_PARTICIONES:
        indice = particiones[nombre]
        assert len(indice) > 0, f"La particion {nombre} esta vacia."
        if fin_anterior is not None:
            assert indice[0] > fin_anterior, (
                f"La particion {nombre} empieza en {indice[0]}, que no es "
                f"posterior al fin de la anterior ({fin_anterior})."
            )
        fin_anterior = indice[-1]

    total = sum(len(particiones[nombre]) for nombre in config.NOMBRES_PARTICIONES)
    assert total == len(etiquetas.index), (
        "Las particiones no cubren exactamente el panel."
    )
