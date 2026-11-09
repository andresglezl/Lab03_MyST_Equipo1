"""Orquestacion de la etapa de desempeno: metricas, atribuciones y figuras.

Reune lo que producen `metrics`, `drawdown`, `attribution` y `report_figures` y
escribe los artifacts que consume el dashboard.
"""

from __future__ import annotations

import pandas as pd

import artifacts_io
import config
from performance import attribution, drawdown, metrics, report_figures


def _agregar_particion(serie: pd.DataFrame, etiquetas: pd.Series) -> pd.DataFrame:
    """Devuelve la serie con la columna de particion garantizada."""
    if "particion" in serie.columns:
        return serie
    return serie.assign(particion=etiquetas.values)


def ejecutar_desempeno(
    panel: pd.DataFrame,
    etiquetas: pd.Series,
    resultados_grupo: list[dict],
    resultado_portafolio: dict,
) -> dict:
    """Calcula metricas y figuras, y escribe los artifacts de 3.7 y 3.10.

    Args:
        panel: panel de precios limpio.
        etiquetas: serie de particiones por fecha.
        resultados_grupo: salidas de `strategy.zscore.procesar_grupo`.
        resultado_portafolio: salida de `backtest.portfolio.ejecutar_portafolio`.

    Returns:
        Diccionario con las metricas, las atribuciones y la lista de figuras.
    """
    consolidado = _agregar_particion(
        resultado_portafolio["consolidado"]["serie"], etiquetas
    )
    pesos_iguales = _agregar_particion(
        resultado_portafolio["pesos_iguales"]["serie"], etiquetas
    )
    individuales = {
        grupo: _agregar_particion(portafolio["serie"], etiquetas)
        for grupo, portafolio in resultado_portafolio["individuales"].items()
    }
    bitacora = resultado_portafolio["bitacora"]
    bitacoras = resultado_portafolio["bitacoras_por_portafolio"]

    # --- Metricas por portafolio y particion ---
    # Cada portafolio usa SU propia bitacora en dolares: el win rate y el profit
    # factor de pesos iguales no se pueden calcular con las operaciones de Risk
    # Parity, porque el P&L de cada operacion depende del capital asignado.
    series_por_portafolio = {
        "consolidado_risk_parity": consolidado,
        "consolidado_pesos_iguales": pesos_iguales,
        **{f"grupo_{grupo}": serie for grupo, serie in individuales.items()},
    }
    tarjetas = {
        nombre: metrics.metricas_por_particion(
            serie, bitacoras[nombre], config.CAPITAL_INICIAL
        )
        for nombre, serie in series_por_portafolio.items()
    }
    artifacts_io.escribir_json(tarjetas, "performance", "metrics.json")

    # --- Sharpe movil de 252 dias de todos los portafolios ---
    movil = pd.DataFrame(
        {
            "consolidado_risk_parity": metrics.sharpe_movil(consolidado["retorno"]),
            "consolidado_pesos_iguales": metrics.sharpe_movil(pesos_iguales["retorno"]),
            **{
                f"grupo_{grupo}": metrics.sharpe_movil(serie["retorno"])
                for grupo, serie in individuales.items()
            },
        }
    )
    movil["particion"] = etiquetas.values
    artifacts_io.escribir_serie(movil, "performance", "rolling_sharpe.parquet")

    # --- Atribucion de costos ---
    costos = attribution.atribucion_de_costos(consolidado)
    artifacts_io.escribir_tabla(costos, "performance", "cost_attribution.csv", indice=False)
    costos_grupo = attribution.atribucion_por_grupo(
        resultado_portafolio["consolidado"]["pnl_bruto_por_grupo"],
        resultado_portafolio["consolidado"]["comision_por_grupo"],
        resultado_portafolio["consolidado"]["prestamo_por_grupo"],
        etiquetas,
    )
    artifacts_io.escribir_tabla(
        costos_grupo, "performance", "cost_attribution_por_grupo.csv", indice=False
    )

    # --- Atribucion del drawdown maximo ---
    ventana, tabla_drawdown = drawdown.atribuir_drawdown(
        consolidado["capital"], resultado_portafolio["consolidado"]["pnl_por_grupo"]
    )
    artifacts_io.escribir_tabla(
        tabla_drawdown, "performance", "drawdown_attribution.csv", indice=False
    )

    # --- Las nueve figuras obligatorias ---
    grupos_tickers = {r["grupo"]: r["tickers"] for r in resultados_grupo}
    senales = {
        r["grupo"]: r["serie_senales"]
        for r in resultados_grupo
    }
    filtros1 = {r["grupo"]: r["serie_f1"] for r in resultados_grupo}
    filtros2 = {r["grupo"]: r["serie_f2"] for r in resultados_grupo}
    moviles = {
        r["grupo"]: artifacts_io.leer_serie("rolling", f"johansen_eigen_{r['grupo']}.parquet")
        for r in resultados_grupo
    }
    umbrales = {r["grupo"]: r["umbrales"].como_dict() for r in resultados_grupo}

    figuras = [
        report_figures.fig01_precios(panel, etiquetas, grupos_tickers),
        report_figures.fig02_spread_senales(senales, etiquetas),
        report_figures.fig03_vector_pesos(filtros1, etiquetas),
        report_figures.fig04_primer_eigenvector(moviles),
        report_figures.fig05_ganancias_kalman(filtros1, filtros2, etiquetas),
        report_figures.fig06_zscore(senales, umbrales, etiquetas),
        report_figures.fig07_curvas_capital(consolidado, pesos_iguales, individuales),
        report_figures.fig08_distribucion_retornos(bitacora),
        report_figures.fig09_risk_parity(
            resultado_portafolio["asignacion"]["tabla"],
            resultado_portafolio["asignacion"]["correlacion"],
        ),
    ]

    resumen = {
        "metricas": {
            portafolio: {
                particion: {
                    clave: valor
                    for clave, valor in tarjeta.items()
                    if clave in ("sharpe", "sortino", "calmar", "max_drawdown",
                                 "win_rate", "profit_factor", "retorno_total",
                                 "capital_final", "n_operaciones")
                }
                for particion, tarjeta in tarjetas_portafolio.items()
            }
            for portafolio, tarjetas_portafolio in tarjetas.items()
        },
        "atribucion_costos_global": costos[costos["particion"] == "global"]
        .to_dict(orient="records")[0],
        "ventana_max_drawdown": {
            "fecha_pico": str(ventana["fecha_pico"].date()),
            "fecha_valle": str(ventana["fecha_valle"].date()),
            "drawdown": ventana["drawdown"],
            "dias_habiles": ventana["dias_habiles"],
        },
        "atribucion_drawdown": tabla_drawdown[
            ["grupo", "pnl_neto_en_ventana", "fraccion_de_la_caida"]
        ].to_dict(orient="records"),
        "figuras": figuras,
        "salidas": [
            "performance/metrics.json",
            "performance/rolling_sharpe.parquet",
            "performance/cost_attribution.csv",
            "performance/cost_attribution_por_grupo.csv",
            "performance/drawdown_attribution.csv",
            *figuras,
        ],
    }
    artifacts_io.actualizar_manifiesto("desempeno", resumen)

    return {
        "tarjetas": tarjetas,
        "costos": costos,
        "costos_por_grupo": costos_grupo,
        "drawdown": tabla_drawdown,
        "ventana_drawdown": ventana,
        "sharpe_movil": movil,
        "figuras": figuras,
        "resumen": resumen,
    }
