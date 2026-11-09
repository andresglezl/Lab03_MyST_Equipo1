"""Modelo de costos: comision y prestamo de valores (seccion 3.1).

Los dos costos se calculan por unidad de NOCIONAL BRUTO desplegado, de modo que
el motor por grupo sea invariante a la escala y el capital se aplique despues.

Comision
--------
0.125% por operacion, por pata, a la entrada y a la salida. Como se cobra sobre
cada cambio de posicion, se implementa sobre el TURNOVER: si el peso objetivo de
una pata pasa de w_{j,t-1} a w_{j,t}, se negocio |w_{j,t} - w_{j,t-1}| de
nocional y se paga la comision sobre eso. Esta formulacion cobra
automaticamente la entrada (de 0 a w), la salida (de w a 0) y cada rebalanceo
intermedio, sin tener que tratarlos como casos distintos.

Prestamo de valores
-------------------
0.25% anual sobre el nocional en corto, con devengo diario en convencion
**actual/365 sobre dias calendario** (convencion C3 del plan de fases):

    costo_t = 0.0025 * (dias_calendario_t / 365) * sum_j max(-w_{j,t-1}, 0)

Se usan dias CALENDARIO entre barras consecutivas, no dias habiles, porque un
corto paga el fin de semana y los feriados. Con nocional constante durante un
ano el costo total es exactamente 0.25%, que es lo que pide el PDF.

Ojo con la convencion de dias: 365 es SOLO para el devengo del prestamo. La
anualizacion de Sharpe, Sortino y el Sharpe movil usa 252 dias habiles
(`config.DIAS_ANUALIZACION`). Son dos cosas distintas: una mide carry de
calendario y la otra escala momentos de retornos observados solo en dias de
operacion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def dias_calendario_entre_barras(indice: pd.DatetimeIndex) -> np.ndarray:
    """Dias calendario transcurridos desde la barra anterior.

    Args:
        indice: indice de fechas del panel.

    Returns:
        Array de la misma longitud; la primera entrada es 1 por convencion.
    """
    diferencias = indice.to_series().diff().dt.days.to_numpy(dtype=float)
    diferencias[0] = 1.0
    return diferencias


def comision_por_turnover(pesos: pd.DataFrame) -> pd.Series:
    """Comision diaria por unidad de nocional bruto, a partir del turnover.

    Args:
        pesos: DataFrame de pesos objetivo por pata (suma de |w| igual a 1 con
            posicion abierta, 0 con posicion cerrada).

    Returns:
        Serie de costo de comision por dia.
    """
    turnover = pesos.diff().abs().sum(axis=1)
    turnover.iloc[0] = pesos.iloc[0].abs().sum()
    return config.COMISION_POR_PATA * turnover


def devengo_prestamo(pesos: pd.DataFrame) -> pd.Series:
    """Costo diario de prestamo por unidad de nocional bruto.

    Se devenga sobre las patas en corto de la posicion que se traia del dia
    anterior, porque es la que estuvo prestada durante el periodo.

    Args:
        pesos: DataFrame de pesos objetivo por pata.

    Returns:
        Serie de costo de prestamo por dia.
    """
    nocional_corto = (-pesos.shift(1)).clip(lower=0.0).sum(axis=1).fillna(0.0)
    fraccion_anual = (
        dias_calendario_entre_barras(pesos.index) / config.DIAS_DEVENGO_PRESTAMO
    )
    return config.TASA_PRESTAMO_ANUAL * pd.Series(
        fraccion_anual, index=pesos.index
    ) * nocional_corto
