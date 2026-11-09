"""Evaluador de un grupo aislado, en dolares (secciones 3.6 y 3.7).

Este modulo NO es un motor aparte: es un envoltorio delgado sobre
`backtest.walkforward.simular_consolidado` que simula un solo grupo con todo el
capital asignado a el. Esa es justamente la razon de que exista.

Por que un solo motor
---------------------
La version anterior tenia dos contabilidades distintas: el grid search y la
comparacion de rebalanceo usaban un motor en "unidades de nocional bruto" con
pesos w_j = beta_j / sum|beta_k|, mientras el portafolio consolidado usaba
acciones y dolares. Esos dos pesos no son el mismo portafolio: beta esta en
ACCIONES, asi que el peso en dolares de la pata j es

    w_j = beta_j P_j / sum_k |beta_k P_k|

y omitir los precios reparte el capital como si todos los activos costaran lo
mismo. La discrepancia no era teorica: la correlacion entre los retornos diarios
de las dos contabilidades era de 0.79 para el grupo de bancos. Es decir que los
umbrales y el ancho de banda se calibraban sobre un portafolio distinto del que
despues se operaba.

Ahora el grid search, la calibracion de la banda y la comparacion diario contra
banda llaman todos a `simular_consolidado`, de modo que lo que se calibra y lo
que se reporta son, por construccion, la misma cosa.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from backtest import walkforward


def simular_grupo_en_dolares(
    precios: pd.DataFrame,
    betas: pd.DataFrame,
    posicion: pd.Series,
    grupo: str,
    ancho_banda: float | None = None,
    capital_inicial: float = config.CAPITAL_INICIAL,
) -> dict:
    """Simula un grupo aislado con todo el capital asignado a el.

    Args:
        precios: panel de precios del grupo.
        betas: DataFrame con columnas `beta_{ticker}` del filtro 1.
        posicion: direccion por dia en {-1, 0, +1}.
        grupo: etiqueta del grupo.
        ancho_banda: None para rebalanceo diario, o el ancho de la banda.
        capital_inicial: capital al inicio de la simulacion.

    Returns:
        La salida de `walkforward.simular_consolidado` para ese unico grupo.
    """
    return walkforward.simular_consolidado(
        {grupo: {"precios": precios, "betas": betas, "posicion": posicion}},
        {grupo: 1.0},
        capital_inicial=capital_inicial,
        ancho_banda=ancho_banda,
        registrar_libro=False,
    )


def sharpe_anualizado(
    retornos: pd.Series, dias_anualizacion: int = config.DIAS_ANUALIZACION
) -> float:
    """Sharpe anualizado de una serie de retornos, con tasa libre de riesgo cero.

    Args:
        retornos: serie de retornos diarios netos.
        dias_anualizacion: dias habiles por ano.

    Returns:
        Sharpe anualizado, o 0.0 si la desviacion estandar es cero.
    """
    desviacion = retornos.std(ddof=1)
    if not np.isfinite(desviacion) or desviacion == 0.0:
        return 0.0
    return float(retornos.mean() / desviacion * np.sqrt(dias_anualizacion))
