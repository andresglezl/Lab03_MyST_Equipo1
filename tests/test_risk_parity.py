"""Prueba obligatoria 4, nivel B: contribuciones al riesgo iguales (3.11.4).

"Nivel B: las contribuciones al riesgo entre grupos son iguales dentro de una
tolerancia declarada."

Tolerancia declarada: `config.TOLERANCIA_RISK_PARITY` = 1e-6 en la contribucion
relativa al riesgo de cada grupo respecto al objetivo 1/G.

Se verifica la solucion de dos maneras independientes: contra el artifact que
escribio el pipeline, y recalculando las contribuciones desde la matriz de
covarianza guardada. Si el solver hubiera escrito pesos que no resuelven el
problema, la segunda verificacion lo detectaria.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import artifacts_io
import config
from backtest import risk_parity


def test_contribuciones_al_riesgo_son_iguales():
    """Cada grupo contribuye 1/G al riesgo, dentro de la tolerancia declarada."""
    tabla = artifacts_io.leer_tabla("riskparity", "weights.csv")
    objetivo = 1.0 / len(tabla)
    desvio = (tabla["contribucion_riesgo_rp"] - objetivo).abs().max()
    assert desvio < config.TOLERANCIA_RISK_PARITY, (
        f"La contribucion al riesgo se desvia de {objetivo:.6f} hasta "
        f"{desvio:.2e}, por encima de la tolerancia "
        f"{config.TOLERANCIA_RISK_PARITY:g}"
    )


def test_contribuciones_recalculadas_desde_la_covarianza():
    """Las contribuciones se reproducen desde la covarianza y los pesos guardados."""
    tabla = artifacts_io.leer_tabla("riskparity", "weights.csv")
    covarianza = pd.read_csv(
        artifacts_io.ruta("riskparity", "spread_cov.csv"), index_col=0
    )

    pesos = tabla.set_index("grupo")["peso_risk_parity"].reindex(covarianza.index)
    contribuciones = risk_parity.contribuciones_al_riesgo(
        pesos.to_numpy(dtype=float), covarianza.to_numpy(dtype=float)
    )
    objetivo = 1.0 / len(covarianza)
    desvio = float(np.max(np.abs(contribuciones - objetivo)))
    assert desvio < config.TOLERANCIA_RISK_PARITY, (
        f"Las contribuciones recalculadas se desvian hasta {desvio:.2e}: "
        f"{dict(zip(covarianza.index, contribuciones.round(10)))}"
    )


def test_pesos_son_positivos_y_suman_uno():
    """Los pesos son una asignacion de capital valida.

    Importa para el limite del 80%: si algun peso fuera negativo, la suma de
    valores absolutos superaria 1 y el nocional bruto desplegado excederia el
    limite sobre el capital consolidado.
    """
    tabla = artifacts_io.leer_tabla("riskparity", "weights.csv")
    pesos = tabla["peso_risk_parity"].to_numpy(dtype=float)
    assert (pesos > 0).all(), f"Hay pesos no positivos: {pesos}"
    assert abs(pesos.sum() - 1.0) < 1e-10, f"Los pesos suman {pesos.sum():.12f}"
    assert abs(np.abs(pesos).sum() - 1.0) < 1e-10


def test_mas_peso_al_grupo_de_menor_volatilidad():
    """Risk Parity asigna mas capital al grupo menos volatil.

    Es la propiedad economica que debe tener la asignacion y sirve de prueba de
    cordura sobre el signo del resultado.
    """
    tabla = artifacts_io.leer_tabla("riskparity", "weights.csv").sort_values(
        "vol_anual_spread"
    )
    pesos = tabla["peso_risk_parity"].to_numpy(dtype=float)
    assert pesos[0] > pesos[-1], (
        "El grupo de menor volatilidad no recibio mas peso: "
        f"{tabla[['grupo', 'vol_anual_spread', 'peso_risk_parity']].to_dict('records')}"
    )


def test_risk_parity_mejora_el_balance_contra_pesos_iguales():
    """Risk Parity reparte el riesgo mas parejo que una asignacion de pesos iguales."""
    tabla = artifacts_io.leer_tabla("riskparity", "weights.csv")
    objetivo = 1.0 / len(tabla)
    desvio_rp = (tabla["contribucion_riesgo_rp"] - objetivo).abs().max()
    desvio_iguales = (tabla["contribucion_riesgo_iguales"] - objetivo).abs().max()
    assert desvio_rp < desvio_iguales, (
        f"Risk Parity (desvio {desvio_rp:.2e}) no mejora sobre pesos iguales "
        f"(desvio {desvio_iguales:.2e})"
    )
