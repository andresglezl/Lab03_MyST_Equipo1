"""Portafolio consolidado en dolares y acciones (secciones 3.1, 3.6 y 3.7).

Es el UNICO motor de backtest del proyecto. El grid search de umbrales, la
calibracion del ancho de banda, la comparacion de politicas de rebalanceo y el
resultado final pasan todos por aqui, a traves de `backtest.engine`, de modo que
lo que se calibra y lo que se reporta son por construccion el mismo portafolio.

Contabilidad diaria
-------------------
Con q_{g,j,t-1} las acciones que se traian y P_{j,t} los precios de hoy:

    pnl_t      = sum_g sum_j q_{g,j,t-1} (P_{j,t} - P_{j,t-1})
    prestamo_t = tasa * (dias_cal_t / 365) * sum_g sum_j max(-q_{g,j,t-1}, 0) P_{j,t-1}
    comision_t = 0.00125 * sum_g sum_j |q_{g,j,t} - q_{g,j,t-1}| P_{j,t}
    C_t        = C_{t-1} + pnl_t - prestamo_t - comision_t

El EFECTIVO se lleva de forma independiente, como un saldo que se mueve con lo
que entra y sale de la cuenta:

    efectivo_t = efectivo_{t-1} - sum_g sum_j (q_{g,j,t} - q_{g,j,t-1}) P_{j,t}
                 - comision_t - prestamo_t

Comprar consume efectivo y vender en corto lo aporta. Llevarlo asi, y no
definirlo como C_t menos el valor de las patas, es lo que hace que la identidad

    C_t = efectivo_t + sum_g sum_j q_{g,j,t} P_{j,t}

sea una verificacion real y no una tautologia: las dos cantidades se calculan
por caminos distintos y la prueba obligatoria 3 comprueba que coincidan.

El walk-forward no reestima nada. Los parametros (beta de Johansen, pata
numeraria, Q, R, vida media, umbrales, ancho de banda y pesos de Risk Parity) se
congelaron en entrenamiento y aqui solo se aplican hacia adelante.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from backtest import costs, rebalance
from strategy import sizing


def simular_consolidado(
    grupos: dict[str, dict],
    pesos: dict[str, float],
    capital_inicial: float = config.CAPITAL_INICIAL,
    ancho_banda: float | None = None,
    registrar_libro: bool = True,
) -> dict:
    """Simula el portafolio consolidado en dolares y acciones.

    Args:
        grupos: diccionario `{etiqueta: {'precios': DataFrame, 'betas': DataFrame,
            'posicion': Series}}`. Todos deben compartir el indice de fechas.
        pesos: peso de capital de cada grupo; las llaves coinciden con `grupos`.
        capital_inicial: capital del portafolio al inicio.
        ancho_banda: None para rebalanceo diario, o el ancho de la banda de
            no-rebalanceo, aplicada sobre acciones.
        registrar_libro: si False omite el libro de patas. Se desactiva durante
            las calibraciones, donde solo interesa la serie agregada y armar el
            libro multiplicaria el costo por el numero de combinaciones.

    Returns:
        Diccionario con 'serie' (DataFrame diario consolidado), 'libro_patas'
        y los P&L por grupo.
    """
    etiquetas = list(grupos)
    indice = grupos[etiquetas[0]]["precios"].index
    dias_calendario = costs.dias_calendario_entre_barras(indice)

    precios = {g: grupos[g]["precios"].to_numpy(dtype=float) for g in etiquetas}
    betas = {g: grupos[g]["betas"].to_numpy(dtype=float) for g in etiquetas}
    direcciones = {g: grupos[g]["posicion"].to_numpy(dtype=int) for g in etiquetas}
    tickers = {
        g: [c.replace("beta_", "") for c in grupos[g]["betas"].columns]
        for g in etiquetas
    }

    acciones_previas = {g: np.zeros(len(tickers[g])) for g in etiquetas}
    capital = capital_inicial
    efectivo = capital_inicial

    n = len(indice)
    registro = {
        clave: np.zeros(n)
        for clave in (
            "capital", "efectivo", "valor_patas", "nocional_bruto", "nocional_corto",
            "pnl_bruto", "comision", "prestamo", "pnl_neto", "exposicion",
            "limite_nocional", "capital_previo", "turnover_usd",
        )
    }
    pnl_grupo = {g: np.zeros(n) for g in etiquetas}
    comision_grupo = {g: np.zeros(n) for g in etiquetas}
    prestamo_grupo = {g: np.zeros(n) for g in etiquetas}
    filas_libro = []

    for t in range(n):
        capital_previo = capital
        pnl_total = comision_total = prestamo_total = 0.0
        flujo_compras = turnover_total = 0.0
        acciones_nuevas = {}

        for g in etiquetas:
            precio_hoy = precios[g][t]
            precio_ayer = precios[g][t - 1] if t > 0 else precio_hoy
            previas = acciones_previas[g]

            # P&L de mantener las acciones que se traian de ayer.
            pnl = float(previas @ (precio_hoy - precio_ayer))
            # Prestamo sobre el nocional corto que se traia.
            corto_previo = float(np.clip(-previas, 0.0, None) @ precio_ayer)
            prestamo = (
                config.TASA_PRESTAMO_ANUAL
                * dias_calendario[t]
                / config.DIAS_DEVENGO_PRESTAMO
                * corto_previo
            )
            # Rebalanceo al cierre de hoy, con el capital del cierre previo.
            objetivo = sizing.acciones_objetivo(
                betas[g][t], precio_hoy, int(direcciones[g][t]), capital_previo, pesos[g]
            )
            if ancho_banda is not None:
                objetivo = rebalance.aplicar_banda_a_acciones(
                    objetivo, previas, precio_hoy, ancho_banda
                )

            negociadas = objetivo - previas
            comision = config.COMISION_POR_PATA * float(np.abs(negociadas) @ precio_hoy)

            pnl_total += pnl
            prestamo_total += prestamo
            comision_total += comision
            flujo_compras += float(negociadas @ precio_hoy)
            turnover_total += float(np.abs(negociadas) @ precio_hoy)
            pnl_grupo[g][t] = pnl
            comision_grupo[g][t] = comision
            prestamo_grupo[g][t] = prestamo
            acciones_nuevas[g] = objetivo

            if registrar_libro:
                for posicion, ticker in enumerate(tickers[g]):
                    filas_libro.append(
                        {
                            "fecha": indice[t],
                            "grupo": g,
                            "ticker": ticker,
                            "acciones": objetivo[posicion],
                            "precio": precio_hoy[posicion],
                            "valor": objetivo[posicion] * precio_hoy[posicion],
                            "acciones_negociadas": negociadas[posicion],
                            "comision": config.COMISION_POR_PATA
                            * abs(negociadas[posicion])
                            * precio_hoy[posicion],
                            "prestamo": config.TASA_PRESTAMO_ANUAL
                            * dias_calendario[t]
                            / config.DIAS_DEVENGO_PRESTAMO
                            * max(-previas[posicion], 0.0)
                            * precio_ayer[posicion],
                        }
                    )

        capital = capital_previo + pnl_total - prestamo_total - comision_total
        # El efectivo se lleva por su propio camino: comprar lo consume, vender
        # en corto lo aporta, y los costos lo reducen.
        efectivo = efectivo - flujo_compras - comision_total - prestamo_total

        valor_patas = sum(float(acciones_nuevas[g] @ precios[g][t]) for g in etiquetas)
        bruto = sum(float(np.abs(acciones_nuevas[g]) @ precios[g][t]) for g in etiquetas)
        corto = sum(
            float(np.clip(-acciones_nuevas[g], 0.0, None) @ precios[g][t])
            for g in etiquetas
        )

        registro["capital"][t] = capital
        registro["efectivo"][t] = efectivo
        registro["valor_patas"][t] = valor_patas
        registro["nocional_bruto"][t] = bruto
        registro["nocional_corto"][t] = corto
        registro["pnl_bruto"][t] = pnl_total
        registro["comision"][t] = comision_total
        registro["prestamo"][t] = prestamo_total
        registro["pnl_neto"][t] = pnl_total - prestamo_total - comision_total
        registro["exposicion"][t] = bruto / capital if capital > 0 else np.nan
        # El limite del 80% se mide contra el capital con el que se dimensiono la
        # posicion, que es el del cierre previo (ver strategy/sizing.py).
        registro["limite_nocional"][t] = config.EXPOSICION_MAXIMA * capital_previo
        registro["capital_previo"][t] = capital_previo
        registro["turnover_usd"][t] = turnover_total

        acciones_previas = acciones_nuevas

    serie = pd.DataFrame(registro, index=indice)
    serie["retorno"] = serie["capital"].pct_change().fillna(
        serie["capital"].iloc[0] / capital_inicial - 1.0
    )
    serie["maximo_historico"] = serie["capital"].cummax()
    serie["drawdown"] = serie["capital"] / serie["maximo_historico"] - 1.0

    pnl_neto_grupo = pd.DataFrame(
        {g: pnl_grupo[g] - comision_grupo[g] - prestamo_grupo[g] for g in etiquetas},
        index=indice,
    )

    return {
        "serie": serie,
        "libro_patas": pd.DataFrame(filas_libro),
        "pnl_por_grupo": pnl_neto_grupo,
        "pnl_bruto_por_grupo": pd.DataFrame(pnl_grupo, index=indice),
        "comision_por_grupo": pd.DataFrame(comision_grupo, index=indice),
        "prestamo_por_grupo": pd.DataFrame(prestamo_grupo, index=indice),
        "capital_inicial": capital_inicial,
    }
