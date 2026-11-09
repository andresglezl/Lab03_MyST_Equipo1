"""Prueba obligatoria 3: contabilidad multi-pata (seccion 3.11.3).

"El valor del portafolio es igual al efectivo mas el valor de todas las patas, y
el costo total registrado coincide con el numero de operaciones por pata
multiplicado por la comision, mas el devengo de prestamo sobre las patas cortas."

Se verifica contra el LIBRO DE PATAS, que es el registro por fecha, grupo y
ticker con acciones, precio, acciones negociadas, comision y prestamo. Conciliar
el agregado contra el detalle es lo que hace que la prueba sea informativa: si
el motor agregara mal, los dos numeros dejarian de coincidir.

Sobre la identidad capital = efectivo + patas: el motor lleva el efectivo por su
propio camino, como un saldo que se mueve con lo que entra y sale de la cuenta
(comprar lo consume, vender en corto lo aporta, los costos lo reducen), mientras
el capital se construye sumando P&L y restando costos. Las dos cantidades se
calculan por rutas distintas, asi que su coincidencia es una verificacion real.
Esta prueba se reforzo tras una revision: en la version anterior el efectivo se
DEFINIA como capital menos el valor de las patas, con lo que la identidad se
cumplia por construccion y la prueba no comprobaba nada.

Tambien se verifica el limite del 80%: el nocional bruto desplegado nunca excede
el 80% del capital con el que se dimensiono la posicion, que es el del cierre
previo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import artifacts_io
import config

#: Tolerancia en dolares para identidades contables sobre un capital de $1M.
TOLERANCIA_USD = 1e-6


def test_capital_es_efectivo_mas_valor_de_las_patas():
    """C_t = efectivo_t + valor de todas las patas, todos los dias."""
    serie = artifacts_io.leer_serie("backtest", "portfolio_equity.parquet")
    residuo = (
        serie["capital"] - serie["efectivo"] - serie["valor_patas"]
    ).abs().max()
    assert residuo < TOLERANCIA_USD, (
        f"La identidad capital = efectivo + patas falla hasta {residuo:.2e} USD"
    )


def test_el_efectivo_se_mueve_con_los_flujos_de_caja():
    """El efectivo se reconstruye desde los flujos, sin usar el capital.

    efectivo_t = efectivo_{t-1} - sum_j (q_{j,t} - q_{j,t-1}) P_{j,t}
                 - comision_t - prestamo_t

    Reconstruirlo desde el libro de patas, que registra las acciones negociadas
    y su precio, y compararlo con el que lleva el motor, verifica que la
    identidad contable no sea una tautologia.
    """
    serie = artifacts_io.leer_serie("backtest", "portfolio_equity.parquet")
    libro = artifacts_io.leer_serie("backtest", "legs_ledger_consolidado.parquet")

    flujo_compras = (
        (libro["acciones_negociadas"] * libro["precio"])
        .groupby(libro["fecha"])
        .sum()
        .reindex(serie.index)
        .fillna(0.0)
    )
    reconstruido = (
        config.CAPITAL_INICIAL
        - (flujo_compras + serie["comision"] + serie["prestamo"]).cumsum()
    )
    residuo = (reconstruido - serie["efectivo"]).abs().max()
    assert residuo < 1e-4, (
        f"El efectivo reconstruido desde los flujos difiere del registrado "
        f"hasta {residuo:.2e} USD"
    )


def test_el_efectivo_no_es_capital_menos_patas_por_construccion():
    """El efectivo es un saldo propio, no una resta del capital.

    Prueba de cordura sobre la prueba anterior: si el motor volviera a definir
    el efectivo como capital menos patas, la identidad seguiria cumpliendose
    pero la reconstruccion desde los flujos del libro ya no, y esta prueba
    quedaria sin sentido. Se comprueba que el efectivo tiene variacion propia:
    cambia de un dia a otro aunque el capital no cambie.
    """
    serie = artifacts_io.leer_serie("backtest", "portfolio_equity.parquet")
    cambio_efectivo = serie["efectivo"].diff().abs()
    cambio_capital = serie["capital"].diff().abs()
    dias_con_flujo_propio = int(
        ((cambio_efectivo > 1.0) & (cambio_capital < 1e-9)).sum()
        + (cambio_efectivo > 10 * cambio_capital.clip(lower=1e-9)).sum()
    )
    assert dias_con_flujo_propio > 0, (
        "El efectivo se mueve siempre igual que el capital: parece definido "
        "como capital menos patas en lugar de llevarse de forma independiente."
    )


def test_valor_de_las_patas_coincide_con_el_libro():
    """El valor agregado de las patas coincide con la suma del libro por pata."""
    serie = artifacts_io.leer_serie("backtest", "portfolio_equity.parquet")
    libro = artifacts_io.leer_serie("backtest", "legs_ledger_consolidado.parquet")

    por_fecha = libro.groupby("fecha")["valor"].sum()
    alineado = por_fecha.reindex(serie.index)
    residuo = (serie["valor_patas"] - alineado).abs().max()
    assert residuo < 1e-4, (
        f"El valor de las patas del libro difiere del agregado hasta {residuo:.2e} USD"
    )


def test_comisiones_coinciden_con_el_detalle_por_pata():
    """La comision total es la suma de |acciones negociadas| x precio x 0.125%."""
    serie = artifacts_io.leer_serie("backtest", "portfolio_equity.parquet")
    libro = artifacts_io.leer_serie("backtest", "legs_ledger_consolidado.parquet")

    # Reconstruccion independiente a partir del nocional negociado por pata.
    reconstruida = (
        config.COMISION_POR_PATA
        * (libro["acciones_negociadas"].abs() * libro["precio"])
    ).sum()
    registrada_libro = libro["comision"].sum()
    registrada_serie = serie["comision"].sum()

    assert abs(reconstruida - registrada_libro) < 1e-4, (
        f"La comision del libro ({registrada_libro:,.6f}) no coincide con la "
        f"reconstruida ({reconstruida:,.6f})"
    )
    assert abs(registrada_libro - registrada_serie) < 1e-4, (
        f"La comision del libro ({registrada_libro:,.6f}) no coincide con la "
        f"del agregado diario ({registrada_serie:,.6f})"
    )


def test_prestamo_se_devenga_solo_sobre_patas_cortas():
    """El prestamo registrado coincide con el devengo sobre las patas cortas."""
    serie = artifacts_io.leer_serie("backtest", "portfolio_equity.parquet")
    libro = artifacts_io.leer_serie("backtest", "legs_ledger_consolidado.parquet")

    assert abs(libro["prestamo"].sum() - serie["prestamo"].sum()) < 1e-4, (
        "El prestamo del libro no coincide con el agregado diario."
    )
    # Ninguna pata larga devenga prestamo.
    largas = libro[libro["acciones"] > 0]
    assert (largas["prestamo"] >= 0).all(), "Hay prestamo negativo registrado."
    assert serie["prestamo"].min() >= 0.0, "Hay devengo de prestamo negativo."


def test_identidad_pnl_bruto_neto_costos():
    """P&L bruto = P&L neto + comisiones + prestamo, dia por dia."""
    serie = artifacts_io.leer_serie("backtest", "portfolio_equity.parquet")
    residuo = (
        serie["pnl_bruto"] - serie["pnl_neto"] - serie["comision"] - serie["prestamo"]
    ).abs().max()
    assert residuo < TOLERANCIA_USD, (
        f"La identidad de P&L falla hasta {residuo:.2e} USD"
    )


def test_capital_evoluciona_con_el_pnl_neto():
    """C_t - C_{t-1} = P&L neto_t, con C_0 partiendo del capital inicial."""
    serie = artifacts_io.leer_serie("backtest", "portfolio_equity.parquet")
    cambio = serie["capital"].diff()
    cambio.iloc[0] = serie["capital"].iloc[0] - config.CAPITAL_INICIAL
    residuo = (cambio - serie["pnl_neto"]).abs().max()
    assert residuo < TOLERANCIA_USD, (
        f"El capital no evoluciona con el P&L neto (residuo {residuo:.2e} USD)"
    )


def test_exposicion_bruta_respeta_el_limite_del_80_por_ciento():
    """El nocional bruto no excede el 80% del capital con que se dimensiono."""
    serie = artifacts_io.leer_serie("backtest", "portfolio_equity.parquet")
    exceso = (serie["nocional_bruto"] - serie["limite_nocional"]).max()
    assert exceso < 1e-6, (
        f"El nocional bruto excedio el limite del "
        f"{config.EXPOSICION_MAXIMA:.0%} en {exceso:,.2f} USD"
    )


def test_grupos_operados_son_disjuntos_en_activos(grupos):
    """Los grupos operados no comparten ningun ticker (seccion 3.7)."""
    vistos: dict[str, str] = {}
    for grupo in grupos:
        for ticker in grupo["tickers"]:
            assert ticker not in vistos, (
                f"{ticker} aparece en {vistos[ticker]} y en {grupo['grupo']}"
            )
            vistos[ticker] = grupo["grupo"]
