"""Paso 3 del embudo: pre-filtro de Engle-Granger (seccion 3.2.3).

Se usa `statsmodels.tsa.stattools.coint`, que corre la regresion OLS del activo
dependiente contra el resto y aplica ADF a los residuales usando los valores
criticos de **MacKinnon para residuales de cointegracion**. Esto importa: los
valores criticos del ADF estandar NO son validos sobre residuales estimados,
porque el OLS ya minimizo su varianza y por tanto los sesga hacia parecer
estacionarios. Usar los criticos equivocados infla los falsos positivos, que es
justo lo que la seccion 3.2 nos pide evitar.

Engle-Granger exige elegir una variable dependiente y su resultado depende de
esa eleccion. Convencion declarada: se prueban las cuatro elecciones posibles y
se reporta el p-value MINIMO junto con la pata que lo produjo. Es el criterio
mas permisivo, apropiado para un PRE-filtro cuyo unico fin es descartar grupos
sin ninguna evidencia; el criterio definitivo es Johansen, que no requiere
elegir variable dependiente.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import coint

import config


def engle_granger_grupo(precios: pd.DataFrame, grupo: tuple[str, ...]) -> dict:
    """Corre Engle-Granger con cada activo del grupo como variable dependiente.

    Args:
        precios: panel de precios restringido a la particion de entrenamiento.
        grupo: tickers del grupo candidato.

    Returns:
        Diccionario con 'eg_pvalue' (el minimo entre las elecciones de pata
        dependiente), 'eg_pata_dependiente', 'eg_estadistico' y
        'pasa_engle_granger'.
    """
    mejor_pvalue = 1.0
    mejor_pata = ""
    mejor_estadistico = 0.0

    for dependiente in grupo:
        regresores = [ticker for ticker in grupo if ticker != dependiente]
        estadistico, pvalue, _ = coint(
            precios[dependiente], precios[regresores], trend="c"
        )
        if pvalue < mejor_pvalue:
            mejor_pvalue, mejor_pata, mejor_estadistico = pvalue, dependiente, estadistico

    return {
        "eg_pvalue": float(mejor_pvalue),
        "eg_pata_dependiente": mejor_pata,
        "eg_estadistico": float(mejor_estadistico),
        "pasa_engle_granger": mejor_pvalue < config.PVALUE_MAXIMO,
    }
