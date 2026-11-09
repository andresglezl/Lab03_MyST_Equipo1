"""Orquestacion del portafolio consolidado de nivel B (seccion 3.7).

Produce los cuatro portafolios que la seccion 3.7 pide comparar:

  1. **Consolidado con Risk Parity**: los dos grupos, con el capital repartido
     por Risk Parity sobre los retornos de los spreads estimados en
     entrenamiento. Es el portafolio principal.
  2. **Consolidado con pesos iguales**: el mismo portafolio con w_g = 1/G, que
     es el benchmark contra el que se mide si Risk Parity aporta algo.
  3. **Cada grupo por separado**: el grupo solo, con todo el 80% del capital
     asignado a el. Sirve para ver si la diversificacion entre grupos ayuda.

El limite del 80% aplica al consolidado, no por grupo: en el portafolio
consolidado los pesos de Risk Parity suman 1, asi que el nocional bruto
desplegado es a lo mas el 80% del capital total, y es menor cuando algun grupo
esta plano.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import artifacts_io
import config
from backtest import risk_parity, walkforward


def _insumos_por_grupo(resultados: list[dict]) -> dict[str, dict]:
    """Arma el diccionario de insumos que consume `simular_consolidado`."""
    return {
        resultado["grupo"]: {
            "precios": resultado["precios"],
            "betas": resultado["serie_f1"][resultado["columnas_beta"]],
            "posicion": resultado["posicion"],
        }
        for resultado in resultados
    }


def bitacora_en_dolares(
    grupo: str, serie_senales: pd.DataFrame, pnl_neto: pd.Series, etiquetas: pd.Series
) -> pd.DataFrame:
    """Convierte las operaciones de un grupo a P&L en dolares del consolidado.

    El P&L por operacion se toma de la contabilidad en dolares del portafolio
    consolidado, acumulando el P&L neto diario del grupo a lo largo de la
    ventana de cada operacion. Asi la bitacora que lee el dashboard es la misma
    que suma a la curva de capital, sin re-escalar nada.

    Args:
        grupo: etiqueta del grupo.
        serie_senales: serie diaria del grupo con 'posicion', 'z' y 'razon_salida'.
        pnl_neto: P&L neto diario del grupo dentro del consolidado, en dolares.
        etiquetas: serie de particiones por fecha.

    Returns:
        DataFrame con una fila por operacion.
    """
    posicion = serie_senales["posicion"].to_numpy()
    n = len(serie_senales)
    filas = []
    t = 0

    while t < n:
        if posicion[t] == 0:
            t += 1
            continue
        inicio = t
        direccion = int(posicion[t])
        while t + 1 < n and posicion[t + 1] == direccion:
            t += 1
        fin = min(t + 1, n - 1)

        tramo = slice(inicio, fin + 1)
        razon = serie_senales["razon_salida"].iloc[fin]
        filas.append(
            {
                "grupo": grupo,
                "direccion": "spread_largo" if direccion == 1 else "spread_corto",
                "fecha_entrada": serie_senales.index[inicio],
                "fecha_salida": serie_senales.index[fin],
                "particion": etiquetas.iloc[inicio],
                "dias_habiles": int(fin - inicio),
                "z_entrada": float(serie_senales["z"].iloc[inicio]),
                "z_salida": float(serie_senales["z"].iloc[fin]),
                "pnl_neto": float(pnl_neto.iloc[tramo].sum()),
                "razon_salida": razon if razon else "abierta",
            }
        )
        t = fin + 1

    tabla = pd.DataFrame(filas)
    if not tabla.empty:
        tabla["ganadora"] = tabla["pnl_neto"] > 0
    return tabla


def ejecutar_portafolio(resultados: list[dict], etiquetas: pd.Series) -> dict:
    """Calcula Risk Parity y simula los cuatro portafolios comparados.

    Args:
        resultados: lista de salidas de `strategy.zscore.procesar_grupo`.
        etiquetas: serie de particiones por fecha.

    Returns:
        Diccionario con los portafolios simulados, la asignacion de Risk Parity
        y el resumen para el manifiesto.
    """
    etiquetas_grupos = [resultado["grupo"] for resultado in resultados]
    insumos = _insumos_por_grupo(resultados)

    # --- Risk Parity sobre los retornos de los spreads, solo entrenamiento ---
    retornos_spread = pd.DataFrame(
        {resultado["grupo"]: resultado["retorno_spread"] for resultado in resultados}
    )
    mascara_entrenamiento = (etiquetas == "entrenamiento").to_numpy()
    operable = np.logical_and.reduce(
        [resultado["serie_f1"]["operable"].to_numpy() for resultado in resultados]
    )
    asignacion = risk_parity.calcular_asignacion(
        retornos_spread.loc[mascara_entrenamiento & operable]
    )

    # --- Portafolios simulados ---
    consolidado = walkforward.simular_consolidado(insumos, asignacion["pesos"])
    pesos_iguales = walkforward.simular_consolidado(
        insumos, asignacion["pesos_iguales"]
    )
    individuales = {
        grupo: walkforward.simular_consolidado({grupo: insumos[grupo]}, {grupo: 1.0})
        for grupo in etiquetas_grupos
    }

    # --- Artifacts ---
    salidas = []
    for nombre, portafolio in (
        ("portfolio_equity.parquet", consolidado),
        ("equity_equal_weight.parquet", pesos_iguales),
    ):
        serie = portafolio["serie"].copy()
        serie["particion"] = etiquetas.values
        artifacts_io.escribir_serie(serie, "backtest", nombre)
        salidas.append(f"backtest/{nombre}")

    bitacoras = []
    for grupo in etiquetas_grupos:
        serie = individuales[grupo]["serie"].copy()
        serie["particion"] = etiquetas.values
        artifacts_io.escribir_serie(serie, "backtest", f"group_equity_{grupo}.parquet")
        salidas.append(f"backtest/group_equity_{grupo}.parquet")

        libro = individuales[grupo]["libro_patas"]
        artifacts_io.escribir_serie(
            libro.reset_index(drop=True), "backtest", f"legs_ledger_{grupo}.parquet"
        )
        salidas.append(f"backtest/legs_ledger_{grupo}.parquet")

        resultado = next(r for r in resultados if r["grupo"] == grupo)
        bitacora = bitacora_en_dolares(
            grupo,
            resultado["serie_senales"],
            consolidado["pnl_por_grupo"][grupo],
            etiquetas,
        )
        artifacts_io.escribir_tabla(
            bitacora, "backtest", f"trades_{grupo}.csv", indice=False
        )
        salidas.append(f"backtest/trades_{grupo}.csv")
        bitacoras.append(bitacora)

    todas = pd.concat(bitacoras, ignore_index=True).sort_values("fecha_entrada")
    artifacts_io.escribir_tabla(todas, "backtest", "trades_all.csv", indice=False)

    # Cada portafolio necesita SU bitacora: el win rate y el profit factor de
    # pesos iguales se calculan con su propio P&L por grupo, no con el de Risk
    # Parity. Los grupos individuales tienen todo el capital asignado a si
    # mismos, asi que su P&L por operacion tampoco coincide con el consolidado.
    bitacoras_por_portafolio = {"consolidado_risk_parity": todas}
    for nombre, portafolio in (
        ("consolidado_pesos_iguales", pesos_iguales),
        *[(f"grupo_{g}", individuales[g]) for g in etiquetas_grupos],
    ):
        partes = []
        for grupo in portafolio["pnl_por_grupo"].columns:
            resultado = next(r for r in resultados if r["grupo"] == grupo)
            partes.append(
                bitacora_en_dolares(
                    grupo,
                    resultado["serie_senales"],
                    portafolio["pnl_por_grupo"][grupo],
                    etiquetas,
                )
            )
        tabla = pd.concat(partes, ignore_index=True).sort_values("fecha_entrada")
        bitacoras_por_portafolio[nombre] = tabla
        artifacts_io.escribir_tabla(
            tabla, "backtest", f"trades_{nombre}.csv", indice=False
        )
        salidas.append(f"backtest/trades_{nombre}.csv")

    libro_consolidado = consolidado["libro_patas"].reset_index(drop=True)
    artifacts_io.escribir_serie(
        libro_consolidado, "backtest", "legs_ledger_consolidado.parquet"
    )

    comparacion = pd.concat(
        [resultado["comparacion_rebalanceo"] for resultado in resultados],
        ignore_index=True,
    )
    artifacts_io.escribir_tabla(
        comparacion, "backtest", "rebalance_comparison.csv", indice=False
    )

    artifacts_io.escribir_tabla(
        asignacion["tabla"], "riskparity", "weights.csv", indice=False
    )
    artifacts_io.escribir_serie(
        retornos_spread.assign(particion=etiquetas.values),
        "riskparity",
        "spread_returns.parquet",
    )
    artifacts_io.escribir_tabla(asignacion["correlacion"], "riskparity", "spread_corr.csv")
    artifacts_io.escribir_tabla(
        asignacion["covarianza"], "riskparity", "spread_cov.csv"
    )
    salidas += [
        "backtest/trades_all.csv",
        "backtest/legs_ledger_consolidado.parquet",
        "backtest/rebalance_comparison.csv",
        "riskparity/weights.csv",
        "riskparity/spread_returns.parquet",
        "riskparity/spread_corr.csv",
        "riskparity/spread_cov.csv",
    ]

    resumen = {
        "grupos": etiquetas_grupos,
        "pesos_risk_parity": {g: float(p) for g, p in asignacion["pesos"].items()},
        "pesos_iguales": {g: float(p) for g, p in asignacion["pesos_iguales"].items()},
        "contribuciones_al_riesgo": dict(
            zip(
                asignacion["tabla"]["grupo"],
                asignacion["tabla"]["contribucion_riesgo_rp"].astype(float),
            )
        ),
        "desvio_maximo_contribuciones": asignacion["desvio_maximo"],
        "tolerancia": asignacion["tolerancia"],
        "contribuciones_iguales": asignacion["contribuciones_iguales"],
        "iteraciones_risk_parity": asignacion["iteraciones"],
        "correlacion_spreads": asignacion["correlacion"].round(6).to_dict(),
        "capital_final_consolidado": float(consolidado["serie"]["capital"].iloc[-1]),
        "capital_final_pesos_iguales": float(pesos_iguales["serie"]["capital"].iloc[-1]),
        "capital_final_por_grupo": {
            g: float(individuales[g]["serie"]["capital"].iloc[-1])
            for g in etiquetas_grupos
        },
        "exposicion_bruta_maxima": float(consolidado["serie"]["exposicion"].max()),
        "salidas": salidas,
    }
    artifacts_io.actualizar_manifiesto("portafolio", resumen)

    return {
        "consolidado": consolidado,
        "pesos_iguales": pesos_iguales,
        "individuales": individuales,
        "asignacion": asignacion,
        "retornos_spread": retornos_spread,
        "bitacora": todas,
        "bitacoras_por_portafolio": bitacoras_por_portafolio,
        "comparacion_rebalanceo": comparacion,
        "resumen": resumen,
    }
