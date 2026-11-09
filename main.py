"""Pipeline completo del Lab 03 MyST - Equipo 1.

Uso:
    python main.py                  # corre el pipeline completo
    python main.py --forzar-descarga  # ignora el cache y vuelve a bajar precios

Las etapas se ejecutan en orden y cada una escribe sus salidas en `artifacts/`.
El dashboard se lanza por separado con `streamlit run dashboard/app.py` y solo
lee esos archivos.
"""

from __future__ import annotations

import argparse

import pandas as pd

import artifacts_io
import config
import numpy as np

from backtest import portfolio
from cointegration import selection
from data import cleaning, loader, splits, universe, validation
from kalman import worked_example
from performance import reporting
from strategy import zscore


def etapa_datos(forzar_descarga: bool = False) -> dict:
    """Ingesta, limpia, valida y particiona el panel de precios.

    Args:
        forzar_descarga: si True ignora el cache local de yfinance.

    Returns:
        Diccionario con 'panel' (DataFrame de precios limpio), 'etiquetas'
        (Serie de particiones) y 'reporte' (validacion de calidad).

    Raises:
        ValueError: si la validacion de datos encuentra fallas fatales.
    """
    universe.verificar_pools_disjuntos()
    tickers = universe.tickers_del_universo()

    crudos = loader.cargar_precios(tickers, forzar_descarga=forzar_descarga)
    panel = cleaning.construir_panel(crudos)

    reporte = validation.validar_panel(panel, crudos)
    return {"panel": panel, "crudos": crudos, "reporte": reporte}


def _guardar_salidas_datos(panel: pd.DataFrame, reporte: dict) -> None:
    """Escribe `artifacts/prices.parquet` y registra la etapa en el manifiesto."""
    etiquetas = splits.etiquetar_particiones(panel.index)
    panel_con_particion = panel.assign(particion=etiquetas)
    artifacts_io.escribir_serie(panel_con_particion, "prices.parquet")

    resumen = splits.resumen_particiones(panel.index)
    artifacts_io.escribir_tabla(resumen, "selection", "particiones.csv")
    artifacts_io.escribir_tabla(reporte["por_ticker"], "selection", "calidad_datos.csv")

    artifacts_io.actualizar_manifiesto(
        "datos",
        {
            "universo": {
                pool.identificador: {
                    "nombre": pool.nombre,
                    "tickers": list(pool.tickers),
                    "tendencia_comun": pool.tendencia_comun,
                    "momento_registro": pool.momento_registro,
                }
                for pool in universe.POOLS_EVALUADOS
            },
            "panel": reporte["resumen"],
            "particiones": {
                nombre: {
                    "inicio": str(fila["inicio"]),
                    "fin": str(fila["fin"]),
                    "n_dias": int(fila["n_dias"]),
                    "porcentaje": float(fila["porcentaje"]),
                }
                for nombre, fila in resumen.iterrows()
            },
            "observaciones": reporte["observaciones"],
            "salidas": ["prices.parquet", "selection/particiones.csv", "selection/calidad_datos.csv"],
        },
    )


def etapa_estrategia(
    panel: pd.DataFrame, etiquetas: pd.Series, resultado_seleccion: dict
) -> list[dict]:
    """Corre filtros, VECM, umbrales y senales para cada grupo seleccionado.

    Args:
        panel: panel de precios limpio, sin la columna de particion.
        etiquetas: serie de particiones por fecha.
        resultado_seleccion: salida de `cointegration.selection.ejecutar_seleccion`.

    Returns:
        Lista con el resultado de `strategy.zscore.procesar_grupo` por grupo.
    """
    resultados = []
    for fila in resultado_seleccion["seleccionados"]:
        tickers = fila["tickers"]
        beta = np.array([fila["_mejor_columna"]["beta_por_ticker"][t] for t in tickers])
        resultados.append(
            zscore.procesar_grupo(
                panel,
                etiquetas,
                tickers,
                fila["pata_numeraria"],
                beta,
                fila["grupo"],
            )
        )
    artifacts_io.actualizar_manifiesto(
        "estrategia", {"grupos": [r["resumen"] for r in resultados]}
    )
    return resultados


def _imprimir_estrategia(grupos: list[dict]) -> None:
    """Imprime el diagnostico de filtros, VECM y umbrales de cada grupo."""
    for resultado in grupos:
        resumen = resultado["resumen"]
        print(f"--- {resumen['grupo']} (numerario {resumen['pata_numeraria']})")
        f1, vm, f2 = resumen["filtro1"], resumen["vecm"], resumen["filtro2"]
        print(
            f"    filtro1: R={f1['R']:.4f} delta={f1['delta_elegido']:g} "
            f"var_innov_norm={f1['diag_var_innovacion_normalizada']:.3f}"
        )
        print(
            f"    VECM   : k={vm['k_rezagos']} beta'alpha={vm['beta_alpha']:+.5f} "
            f"vida_media={vm['vida_media_dias']:.1f}d "
            f"LjungBox({vm['ljungbox_rezagos']}) p={vm['ljungbox_pvalue']:.4f} "
            f"blancos={vm['residuales_blancos']}"
        )
        print(
            f"    filtro2: ventana_obj={f2['ventana_efectiva_objetivo']:.1f}d "
            f"K_inf={f2['ganancia_estacionaria']:.5f} "
            f"var_innov_norm={f2['diag_var_innovacion_normalizada']:.3f} "
            f"z_sd={f2['diag_z_desviacion']:.3f}"
        )
        print(
            f"    umbrales: {resumen['umbrales']} "
            f"banda={resumen['ancho_banda_calibrado']}"
        )
        print(
            f"    senales: {resumen['n_operaciones']} operaciones, "
            f"{resumen['dias_en_mercado']} dias en mercado"
        )


def _imprimir_desempeno(resultado: dict) -> None:
    """Imprime la tarjeta de desempeno, la atribucion de costos y el drawdown."""
    filas = []
    for portafolio, tarjetas in resultado["tarjetas"].items():
        for particion, tarjeta in tarjetas.items():
            filas.append(
                {
                    "portafolio": portafolio,
                    "particion": particion,
                    "sharpe": round(tarjeta["sharpe"], 3),
                    "sortino": round(tarjeta["sortino"], 3),
                    "calmar": round(tarjeta["calmar"], 3),
                    "max_dd": round(tarjeta["max_drawdown"], 4),
                    "win_rate": round(tarjeta["win_rate"], 3),
                    "profit_factor": round(tarjeta["profit_factor"], 3),
                    "ret_total": round(tarjeta["retorno_total"], 4),
                    "n_ops": tarjeta["n_operaciones"],
                }
            )
    print(pd.DataFrame(filas).to_string(index=False))
    print()
    print("Atribucion de costos (USD):")
    print(resultado["costos"].to_string(index=False))
    print()
    ventana = resultado["ventana_drawdown"]
    print(
        f"Drawdown maximo del consolidado: {ventana['drawdown']:.2%} "
        f"del {ventana['fecha_pico'].date()} al {ventana['fecha_valle'].date()} "
        f"({ventana['dias_habiles']} dias habiles)"
    )
    print(
        resultado["drawdown"][
            ["grupo", "pnl_neto_en_ventana", "fraccion_de_la_caida"]
        ].to_string(index=False)
    )
    print()
    print(f"Figuras generadas: {len(resultado['figuras'])}")
    for figura in resultado["figuras"]:
        print(f"  {figura}")


def _imprimir_portafolio(resultado: dict) -> None:
    """Imprime los pesos de Risk Parity, su verificacion y los capitales finales."""
    asignacion = resultado["asignacion"]
    print(asignacion["tabla"].to_string(index=False))
    print()
    print("Correlacion entre spreads:")
    print(asignacion["correlacion"].round(4).to_string())
    print()
    resumen = resultado["resumen"]
    print(
        f"Contribuciones al riesgo iguales dentro de "
        f"{resumen['tolerancia']:g}: {resumen['contribuciones_iguales']} "
        f"(desvio maximo {resumen['desvio_maximo_contribuciones']:.2e}, "
        f"{resumen['iteraciones_risk_parity']} iteraciones)"
    )
    print(f"Exposicion bruta maxima: {resumen['exposicion_bruta_maxima']:.1%}")
    print()
    print("Capital final:")
    print(f"  consolidado Risk Parity: ${resumen['capital_final_consolidado']:,.0f}")
    print(f"  consolidado pesos iguales: ${resumen['capital_final_pesos_iguales']:,.0f}")
    for grupo, capital in resumen["capital_final_por_grupo"].items():
        print(f"  {grupo} solo: ${capital:,.0f}")
    print()
    print("Rebalanceo diario contra banda de no-rebalanceo:")
    columnas = [
        "grupo", "politica", "ancho_banda", "particion", "turnover_usd",
        "comision_total", "pnl_neto", "sharpe_neto", "dias_con_operacion",
    ]
    print(resultado["comparacion_rebalanceo"][columnas].round(4).to_string(index=False))


def _imprimir_seleccion(resultado: dict) -> None:
    """Imprime la tabla de grupos probados y la defensa de multiplicidad."""
    columnas = [
        "pool", "grupo", "corr_niveles_min", "eg_pvalue", "johansen_traza_r0",
        "johansen_traza_critico_95", "johansen_rango", "adf_pvalue",
        "vida_media_dias", "etapa_descarte", "operado",
    ]
    print(resultado["tabla"][columnas].to_string(index=False))
    resumen = resultado["resumen"]
    print()
    print(f"Grupos evaluados (m): {resumen['m_grupos_evaluados']}")
    print(
        f"Falsos positivos esperados a alpha={resumen['alpha']}: "
        f"{resumen['falsos_positivos_esperados']}"
    )
    print(f"Grupos que pasaron el embudo completo: {resumen['n_grupos_que_pasaron']}")
    print(f"Descartes por etapa: {resumen['descartes_por_etapa']}")
    print()
    for grupo in resumen["grupos_operados"]:
        print(
            f"  OPERADO {grupo['pool']} {grupo['grupo']}: "
            f"numerario={grupo['pata_numeraria']} beta={grupo['beta']} "
            f"adf_p={grupo['adf_pvalue']} vida_media={grupo['vida_media_dias']}d"
        )


def main() -> None:
    """Punto de entrada del pipeline."""
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument(
        "--forzar-descarga",
        action="store_true",
        help="Ignora el cache local y vuelve a descargar los precios.",
    )
    argumentos = analizador.parse_args()

    config.fijar_semilla()
    config.preparar_directorios()
    artifacts_io.inicializar_manifiesto()

    print("=" * 78)
    print("ETAPA 1 - DATOS")
    print("=" * 78)
    salida = etapa_datos(forzar_descarga=argumentos.forzar_descarga)
    validation.exigir_panel_valido(salida["reporte"])
    print(validation.formatear_reporte(salida["reporte"]))
    print()
    print("Particion cronologica 60/20/20:")
    print(splits.resumen_particiones(salida["panel"].index).to_string())
    _guardar_salidas_datos(salida["panel"], salida["reporte"])
    print()
    print("=" * 78)
    print("ETAPA 2 - COINTEGRACION")
    print("=" * 78)
    entrenamiento = splits.recortar_a_particion(salida["panel"], "entrenamiento")
    resultado_seleccion = selection.ejecutar_seleccion(entrenamiento)
    _imprimir_seleccion(resultado_seleccion)

    print()
    print("=" * 78)
    print("ETAPA 3 - FILTROS DE KALMAN, VECM Y SENALES")
    print("=" * 78)
    etiquetas = splits.etiquetar_particiones(salida["panel"].index)
    grupos = etapa_estrategia(salida["panel"], etiquetas, resultado_seleccion)
    _imprimir_estrategia(grupos)

    print()
    print("=" * 78)
    print("ETAPA 4 - PORTAFOLIO CONSOLIDADO Y RISK PARITY")
    print("=" * 78)
    resultado_portafolio = portfolio.ejecutar_portafolio(grupos, etiquetas)
    _imprimir_portafolio(resultado_portafolio)

    print()
    print("=" * 78)
    print("ETAPA 5 - DESEMPENO, ATRIBUCIONES Y FIGURAS")
    print("=" * 78)
    resultado_desempeno = reporting.ejecutar_desempeno(
        salida["panel"], etiquetas, grupos, resultado_portafolio
    )
    _imprimir_desempeno(resultado_desempeno)

    ejemplos = worked_example.generar_y_guardar(salida["panel"], etiquetas, grupos)
    artifacts_io.actualizar_manifiesto("ejemplo_numerico", {"salidas": ejemplos})
    print()
    print(f"Ejemplos numericos de Powell (seccion 3.5): {len(ejemplos)} archivos")

    print()
    print(f"Artifacts escritos en: {config.DIR_ARTIFACTS}")


if __name__ == "__main__":
    main()
