"""Orquestacion por grupo: filtros, VECM, z-score, umbrales y senales (3.4 y 3.6).

De donde sale el z-score de senal (convencion C5 del plan de fases)
-------------------------------------------------------------------
La seccion 3.6 pide que "el z-score de generacion de senales se derive de los
residuales del VECM, normalizados con las estimaciones en tiempo real del filtro
2". Nuestra interpretacion declarada es que el residual relevante es el TERMINO
DE CORRECCION DE ERROR, es decir la desviacion del equilibrio ECT_t = beta' P_t,
y no el residual de pronostico a un paso de las ecuaciones en diferencias.

Por que es consistente con la seccion 3.4, que es la que define el filtro 2:
3.4 dice que el filtro 2 observa "el spread realizado S_t" y produce
z_t = (S_t - mu_hat_t) / sigma_hat_t con estadisticos previos a la
actualizacion. El ECT ES el spread realizado: es la misma cantidad
beta' P_t que el filtro 1 produce como su innovacion. Si en cambio
alimentaramos al filtro 2 con el residual de pronostico de Delta P, el z-score
mediria "cuanto se movio hoy el precio frente a lo pronosticado", que es una
sorpresa de un dia y no una desviacion del equilibrio, y la regla de trading de
3.6 (entrar cuando el portafolio esta por debajo de su equilibrio) dejaria de
tener sentido.

El VECM no queda de adorno: aporta (i) la verificacion de que beta'alpha < 0, es
decir que el desequilibrio efectivamente se corrige, (ii) la vida media
implicita que calibra el filtro 2 y el periodo maximo de tenencia, y (iii) el
diagnostico de blancura de sus residuales, que es la prueba de que el modelo
dinamico es adecuado. Esos residuales de pronostico se reportan como diagnostico
en `artifacts/strategy/`, no como insumo de la senal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import artifacts_io
import config
from backtest import engine, rebalance, risk_parity
from cointegration import rolling
from data import splits
from kalman import filter1_weights, filter2_signal
from strategy import signals, thresholds, vecm


def procesar_grupo(
    panel: pd.DataFrame,
    etiquetas: pd.Series,
    tickers: list[str],
    pata_numeraria: str,
    beta_entrenamiento: np.ndarray,
    grupo: str,
) -> dict:
    """Corre el pipeline completo de un grupo, de los filtros a las senales.

    Todo lo que se ESTIMA usa solo entrenamiento: la calibracion de Q y R, el
    VECM, la vida media que calibra el filtro 2 y el grid search de umbrales. La
    recursion de los filtros recorre toda la muestra porque es causal.

    Args:
        panel: panel de precios completo.
        etiquetas: serie de particiones por fecha.
        tickers: activos del grupo en el orden del vector beta.
        pata_numeraria: ticker numerario congelado desde entrenamiento.
        beta_entrenamiento: vector cointegrante normalizado de entrenamiento.
        grupo: etiqueta del grupo para nombres de archivo.

    Returns:
        Diccionario con las series de los filtros, el resultado del VECM, los
        umbrales elegidos, la simulacion del grupo y el resumen para el manifiesto.
    """
    precios = panel[tickers]
    mascara_entrenamiento = (etiquetas == "entrenamiento").to_numpy()
    precios_entrenamiento = precios.loc[mascara_entrenamiento]

    # --- Filtro 1: pesos dinamicos ---
    resultado_f1 = filter1_weights.ejecutar_filtro1(
        precios, precios_entrenamiento, tickers, pata_numeraria, beta_entrenamiento
    )
    serie_f1 = resultado_f1["serie"]
    columnas_beta = [f"beta_{ticker}" for ticker in tickers]

    # --- VECM sobre entrenamiento: rezagos, beta'alpha y vida media ---
    ect_entrenamiento = vecm.calcular_ect(
        precios_entrenamiento, tickers, beta_entrenamiento
    )
    k_rezagos, tabla_rezagos = vecm.seleccionar_rezagos(ect_entrenamiento)
    resultado_vecm = vecm.estimar_vecm(
        precios_entrenamiento, tickers, beta_entrenamiento, k_rezagos
    )
    vida_media = resultado_vecm["reducida_ect"]["vida_media_dias"]

    # --- Filtro 2: media y varianza latentes del spread, z-score ex-ante ---
    spread = serie_f1["spread"]
    operable = serie_f1["operable"]
    spread_entrenamiento_operable = spread.loc[
        mascara_entrenamiento & operable.to_numpy()
    ]
    resultado_f2 = filter2_signal.ejecutar_filtro2(
        spread,
        vida_media_entrenamiento=vida_media,
        varianza_semilla=float(spread_entrenamiento_operable.var(ddof=1)),
    )
    serie_f2 = resultado_f2["serie"]
    z = serie_f2["z"]

    # --- Grid search de umbrales, solo en entrenamiento ---
    umbrales, tabla_malla = thresholds.buscar_umbrales(
        precios_entrenamiento,
        serie_f1.loc[mascara_entrenamiento, columnas_beta],
        z.loc[mascara_entrenamiento],
        operable.loc[mascara_entrenamiento],
        vida_media_dias=vida_media,
        grupo=grupo,
    )

    # --- Senales sobre la muestra completa con los umbrales congelados ---
    tabla_senales = signals.generar_senales(z, operable, umbrales)
    posicion = tabla_senales["posicion"]

    # --- Ancho de banda de no-rebalanceo, calibrado solo en entrenamiento ---
    ancho_banda, tabla_banda = rebalance.calibrar_ancho_banda(
        precios_entrenamiento,
        serie_f1.loc[mascara_entrenamiento, columnas_beta],
        posicion.loc[mascara_entrenamiento],
        grupo,
    )

    # --- Rebalanceo diario contra banda, con el mismo motor en dolares ---
    # La politica que opera la estrategia es el rebalanceo DIARIO (seccion 3.6);
    # la banda se simula en paralelo solo para cuantificar su costo relativo.
    comparacion_rebalanceo = rebalance.comparar_politicas(
        precios, serie_f1[columnas_beta], posicion, grupo, ancho_banda, etiquetas
    )

    # --- Serie diaria de senales, para el dashboard y las figuras ---
    serie_senales = tabla_senales.copy()
    serie_senales["z"] = z
    serie_senales["particion"] = etiquetas.values
    pesos = signals.pesos_objetivo(serie_f1[columnas_beta], precios, posicion)
    for columna in pesos.columns:
        serie_senales[columna] = pesos[columna]

    # --- Retorno del spread por unidad de nocional bruto, para Risk Parity ---
    retorno_spread = risk_parity.retorno_spread(
        spread, serie_f1[columnas_beta], precios
    )

    # --- Johansen movil para la figura 4 ---
    tabla_movil = rolling.johansen_movil(precios, tickers, pata_numeraria)

    salidas = []
    salidas += filter1_weights.guardar_filtro1(grupo, resultado_f1)
    salidas += filter2_signal.guardar_filtro2(grupo, resultado_f2)
    salidas += rolling.guardar_johansen_movil(grupo, tabla_movil)

    serie_senales = serie_senales.join(
        serie_f2[["spread", "mu_predicha", "sigma_predicha", "ganancia_filtro2"]]
    )
    artifacts_io.escribir_serie(serie_senales, "strategy", f"signals_{grupo}.parquet")
    artifacts_io.escribir_tabla(
        tabla_malla, "strategy", f"grid_search_{grupo}.csv", indice=False
    )
    artifacts_io.escribir_tabla(
        tabla_rezagos, "strategy", f"vecm_rezagos_{grupo}.csv", indice=False
    )
    artifacts_io.escribir_tabla(
        vecm.tabla_alphas(resultado_vecm, tickers),
        "strategy",
        f"vecm_alphas_{grupo}.csv",
        indice=False,
    )
    artifacts_io.escribir_tabla(
        tabla_banda, "strategy", f"calibracion_banda_{grupo}.csv", indice=False
    )
    salidas += [
        f"strategy/signals_{grupo}.parquet",
        f"strategy/grid_search_{grupo}.csv",
        f"strategy/vecm_rezagos_{grupo}.csv",
        f"strategy/vecm_alphas_{grupo}.csv",
        f"strategy/calibracion_banda_{grupo}.csv",
    ]

    resumen = {
        "grupo": grupo,
        "tickers": tickers,
        "pata_numeraria": pata_numeraria,
        "filtro1": {
            "R": resultado_f1["calibracion"]["R"],
            "delta_elegido": resultado_f1["calibracion"]["delta_elegido"],
            "burn_in_dias": config.BURN_IN_DIAS,
            **{f"diag_{k}": v for k, v in resultado_f1["diagnostico"].items()},
        },
        "vecm": {
            "k_rezagos": k_rezagos,
            "beta_alpha": resultado_vecm["beta_alpha"],
            "revierte": resultado_vecm["revierte"],
            "phi_ect": resultado_vecm["reducida_ect"]["phi"],
            "vida_media_dias": vida_media,
            "ljungbox_pvalue": resultado_vecm["reducida_ect"]["ljungbox_pvalue"],
            "ljungbox_rezagos": resultado_vecm["reducida_ect"]["ljungbox_rezagos"],
            "residuales_blancos": resultado_vecm["reducida_ect"]["residuales_blancos"],
        },
        "filtro2": {
            **{
                k: v
                for k, v in resultado_f2["calibracion"].items()
                if isinstance(v, (int, float))
            },
            **{f"diag_{k}": v for k, v in resultado_f2["diagnostico"].items()},
        },
        "umbrales": umbrales.como_dict(),
        "ancho_banda_calibrado": ancho_banda,
        "n_operaciones": thresholds.contar_operaciones(posicion),
        "dias_en_mercado": int((posicion != 0).sum()),
        "salidas": salidas,
    }

    return {
        "grupo": grupo,
        "tickers": tickers,
        "precios": precios,
        "beta_johansen": beta_entrenamiento,
        "serie_f1": serie_f1,
        "serie_f2": serie_f2,
        "vecm": resultado_vecm,
        "umbrales": umbrales,
        "ancho_banda": ancho_banda,
        "posicion": posicion,
        "serie_senales": serie_senales,
        "comparacion_rebalanceo": comparacion_rebalanceo,
        "retorno_spread": retorno_spread,
        "columnas_beta": columnas_beta,
        "resumen": resumen,
    }
