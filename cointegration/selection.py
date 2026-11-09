"""Paso 6: orquestacion del embudo y documentacion de la seleccion (3.2.6).

El embudo se aplica a TODOS los grupos candidatos de TODOS los pools, siempre,
y cada grupo se etiqueta con la etapa en la que fue descartado. Un grupo
descartado en una etapa temprana igual recorre las etapas posteriores: cuesta
poco y permite reportar la tabla completa, que es lo que pide la seccion 3.2.6
y lo que sostiene el argumento de pruebas multiples.

Criterio de seleccion final, declarado antes de ver resultados: se opera el
grupo que pasa las CUATRO etapas (correlacion, Engle-Granger, Johansen al 95% y
ADF del spread) con vida media operable, y si hay varios en un mismo pool, el
de menor p-value de ADF. Los grupos operados deben ser disjuntos en activos,
condicion que se satisface automaticamente porque los pools son disjuntos y se
toma a lo mas un grupo por pool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import artifacts_io
import config
from cointegration import engle_granger, johansen, screening, spread
from data import universe

#: Etapas del embudo en orden. 'operado' significa que el grupo paso todas.
ETAPAS = ("correlacion", "engle_granger", "johansen", "adf_spread", "operado")


def _etapa_de_descarte(fila: dict) -> str:
    """Determina en que etapa del embudo se descarto un grupo.

    Args:
        fila: diccionario con las banderas 'pasa_*' del grupo.

    Returns:
        Nombre de la etapa que el grupo no supero, o 'operado' si paso todas.
    """
    if not fila["pasa_correlacion"]:
        return "correlacion"
    if not fila["pasa_engle_granger"]:
        return "engle_granger"
    if not fila["pasa_johansen"]:
        return "johansen"
    if not (fila["pasa_adf_spread"] and fila["vida_media_operable"]):
        return "adf_spread"
    return "operado"


def evaluar_grupo(
    precios_entrenamiento: pd.DataFrame,
    pool: universe.PoolCandidatos,
    grupo: tuple[str, ...],
) -> dict:
    """Corre el embudo completo sobre un grupo candidato.

    Args:
        precios_entrenamiento: panel restringido a la particion de entrenamiento.
        pool: pool al que pertenece el grupo.
        grupo: tickers del grupo candidato.

    Returns:
        Diccionario con todos los estadisticos del grupo, la etapa de descarte,
        el vector normalizado elegido y la tabla comparativa de columnas.
    """
    resumen_correlacion = screening.aplicar_filtro_correlacion(
        precios_entrenamiento, grupo
    )
    resumen_eg = engle_granger.engle_granger_grupo(precios_entrenamiento, grupo)
    resultado_johansen = johansen.prueba_johansen(precios_entrenamiento, grupo)
    mejor_columna, tabla_columnas = spread.seleccionar_columna_mas_reversiva(
        precios_entrenamiento, resultado_johansen
    )

    fila = {
        "pool": pool.identificador,
        "grupo": "-".join(grupo),
        "tickers": list(grupo),
        **resumen_correlacion,
        **resumen_eg,
        "johansen_traza_r0": resultado_johansen["traza_r0"],
        "johansen_traza_critico_95": resultado_johansen["traza_critico_r0"],
        "johansen_maxeig_r0": resultado_johansen["maxeig_r0"],
        "johansen_maxeig_critico_95": resultado_johansen["maxeig_critico_r0"],
        "johansen_rango": resultado_johansen["rango"],
        "pasa_johansen": resultado_johansen["pasa_johansen"],
        "columna_elegida": mejor_columna["columna"],
        "pata_numeraria": mejor_columna["pata_numeraria"],
        "adf_estadistico": mejor_columna["adf_estadistico"],
        "adf_pvalue": mejor_columna["adf_pvalue"],
        "vida_media_dias": mejor_columna["vida_media_dias"],
        "pasa_adf_spread": mejor_columna["pasa_adf_spread"],
        "vida_media_operable": mejor_columna["vida_media_operable"],
    }
    fila["etapa_descarte"] = _etapa_de_descarte(fila)
    fila["_johansen"] = resultado_johansen
    fila["_mejor_columna"] = mejor_columna
    fila["_tabla_columnas"] = tabla_columnas
    return fila


def ejecutar_embudo(precios_entrenamiento: pd.DataFrame) -> list[dict]:
    """Evalua todos los grupos candidatos de todos los pools del universo.

    Args:
        precios_entrenamiento: panel restringido a la particion de entrenamiento.

    Returns:
        Lista de filas, una por grupo candidato, en orden determinista.
    """
    filas = []
    for pool in universe.POOLS_EVALUADOS:
        for grupo in screening.generar_grupos_candidatos(pool):
            filas.append(evaluar_grupo(precios_entrenamiento, pool, grupo))
    return filas


def seleccionar_grupos_operados(filas: list[dict]) -> list[dict]:
    """Elige los grupos a operar: a lo mas uno por pool, el mejor que paso todo.

    Args:
        filas: salida de `ejecutar_embudo`.

    Returns:
        Lista de las filas seleccionadas, ordenada por identificador de pool.
        Puede tener menos elementos que `config.N_GRUPOS_OBJETIVO` si no hay
        suficientes grupos cointegrados; en ese caso el pipeline continua con
        los que haya y la limitacion se declara en el reporte.
    """
    aprobados = [fila for fila in filas if fila["etapa_descarte"] == "operado"]
    seleccionados = []
    for pool in universe.POOLS_EVALUADOS:
        del_pool = [f for f in aprobados if f["pool"] == pool.identificador]
        if del_pool:
            seleccionados.append(min(del_pool, key=lambda f: f["adf_pvalue"]))
    return seleccionados


def _verificar_disjuntos(seleccionados: list[dict]) -> None:
    """Valida que los grupos operados no compartan activos (seccion 3.7).

    Raises:
        ValueError: si dos grupos operados comparten algun ticker.
    """
    vistos: dict[str, str] = {}
    for fila in seleccionados:
        for ticker in fila["tickers"]:
            if ticker in vistos:
                raise ValueError(
                    f"Los grupos operados {vistos[ticker]} y {fila['grupo']} "
                    f"comparten {ticker}: deben ser disjuntos."
                )
            vistos[ticker] = fila["grupo"]


def _tabla_grupos_probados(filas: list[dict], seleccionados: list[dict]) -> pd.DataFrame:
    """Construye la tabla de todos los grupos probados para `artifacts/`."""
    operados = {fila["grupo"] for fila in seleccionados}
    columnas = [
        "pool", "grupo", "momento_registro_pool", "corr_niveles_min",
        "corr_retornos_min", "par_mas_debil", "pasa_correlacion", "eg_pvalue",
        "eg_pata_dependiente", "pasa_engle_granger", "johansen_traza_r0",
        "johansen_traza_critico_95", "johansen_maxeig_r0",
        "johansen_maxeig_critico_95", "johansen_rango", "pasa_johansen",
        "columna_elegida", "pata_numeraria", "adf_estadistico", "adf_pvalue",
        "pasa_adf_spread", "vida_media_dias", "vida_media_operable",
        "etapa_descarte", "operado",
    ]
    registros = []
    for fila in filas:
        pool = universe.pool_por_identificador(fila["pool"])
        registro = {
            clave: fila.get(clave)
            for clave in columnas
            if clave not in ("momento_registro_pool", "operado")
        }
        registro["momento_registro_pool"] = pool.momento_registro
        registro["operado"] = fila["grupo"] in operados
        registros.append(registro)

    tabla = pd.DataFrame(registros)[columnas]
    numericas = tabla.select_dtypes(include=[float]).columns
    tabla[numericas] = tabla[numericas].round(6)
    return tabla.sort_values(["pool", "adf_pvalue"]).reset_index(drop=True)


def ejecutar_seleccion(precios_entrenamiento: pd.DataFrame) -> dict:
    """Corre el embudo, selecciona los grupos y escribe los artifacts de 3.2.

    Args:
        precios_entrenamiento: panel restringido a la particion de entrenamiento.

    Returns:
        Diccionario con 'filas' (todos los grupos probados), 'seleccionados'
        (los grupos a operar) y 'resumen' (conteos y defensa de multiplicidad).
    """
    filas = ejecutar_embudo(precios_entrenamiento)
    seleccionados = seleccionar_grupos_operados(filas)
    _verificar_disjuntos(seleccionados)

    tabla = _tabla_grupos_probados(filas, seleccionados)
    artifacts_io.escribir_tabla(tabla, "selection", "groups_tested.csv", indice=False)

    for fila in seleccionados:
        etiqueta = fila["grupo"]
        artifacts_io.escribir_json(
            johansen.resumen_johansen_para_artifact(fila["_johansen"]),
            "selection",
            f"johansen_{etiqueta}.json",
        )
        artifacts_io.escribir_tabla(
            fila["_tabla_columnas"],
            "selection",
            f"vector_comparison_{etiqueta}.csv",
            indice=False,
        )

    m = len(filas)
    resumen = {
        "m_grupos_evaluados": m,
        "alpha": config.PVALUE_MAXIMO,
        "falsos_positivos_esperados": round(config.PVALUE_MAXIMO * m, 2),
        "n_grupos_que_pasaron": int((tabla["etapa_descarte"] == "operado").sum()),
        "descartes_por_etapa": tabla["etapa_descarte"].value_counts().to_dict(),
        "grupos_operados": [
            {
                "pool": fila["pool"],
                "grupo": fila["grupo"],
                "tickers": fila["tickers"],
                "pata_numeraria": fila["pata_numeraria"],
                "beta": {
                    ticker: float(np.round(valor, 6))
                    for ticker, valor in fila["_mejor_columna"]["beta_por_ticker"].items()
                },
                "adf_pvalue": round(fila["adf_pvalue"], 6),
                "vida_media_dias": round(fila["vida_media_dias"], 2),
                "johansen_rango": fila["johansen_rango"],
            }
            for fila in seleccionados
        ],
        "salidas": [
            "selection/groups_tested.csv",
            *[f"selection/johansen_{f['grupo']}.json" for f in seleccionados],
            *[f"selection/vector_comparison_{f['grupo']}.csv" for f in seleccionados],
        ],
    }
    artifacts_io.actualizar_manifiesto("cointegracion", resumen)
    return {"filas": filas, "seleccionados": seleccionados, "resumen": resumen, "tabla": tabla}
