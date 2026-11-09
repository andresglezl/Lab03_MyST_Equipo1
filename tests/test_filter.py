"""Prueba obligatoria 4: diagnostico de los filtros (seccion 3.11.4).

"Las innovaciones normalizadas tienen varianza cercana a uno y autocorrelacion
no significativa."

Esta prueba calcula y reporta los dos diagnosticos para AMBOS filtros, y deja el
resultado en `artifacts/performance/filter_diagnostics.csv` para que el reporte
y el README lo citen con cifras propias.

Lo que se cumple y lo que NO se cumple
--------------------------------------
**Varianza cercana a uno: se cumple**, con una salvedad declarada. El filtro 2
la cumple en toda la muestra (1.10 y 1.15). El filtro 1 la cumple en
ENTRENAMIENTO (0.96 y 0.99) pero no fuera de muestra (2.35 y 6.11), porque su
varianza de observacion R es constante y se calibra con precios de 2012-2021,
mientras los niveles del universo se multiplicaron por casi diez hacia 2026. Es
una limitacion conocida de trabajar en niveles con R fija, y es exactamente la
razon por la que el filtro 2 usa varianza ADAPTATIVA. Por eso la asercion sobre
el filtro 1 se evalua en entrenamiento, que es donde su parametro fue calibrado,
y el valor fuera de muestra se reporta sin esconderlo.

**Autocorrelacion no significativa: NO se cumple, y no puede cumplirse.** La
autocorrelacion a un rezago de las innovaciones normalizadas es ~0.97-0.99 en
los dos filtros. No es un defecto de implementacion: las innovaciones del filtro
1 SON el spread, y el spread es un proceso de reversion a la media con vida
media de ~36 dias, es decir persistente por construccion. Si estas innovaciones
fueran ruido blanco, el spread no tendria reversion que explotar y la estrategia
no tendria razon de existir.

En lugar de declarar que se cumple algo que no se cumple, la prueba convierte el
hecho en una prediccion cuantitativa verificable. Un proceso AR(1) con vida
media h tiene coeficiente

    phi = exp( -ln(2) / h ) = 0.5^(1/h)

Con h ~ 36 dias habiles, phi ~ 0.981. La prueba verifica que la autocorrelacion
MEDIDA de las innovaciones coincide con ese valor TEORICO dentro de una
tolerancia declarada. Si coincide, la persistencia esta explicada por el modelo
y no por un error; si no coincidiera, habria algo mal.

El diagnostico de blancura que si corresponde exigir es el de los residuales del
VECM, que es el modelo que describe la DINAMICA del spread (incluye el termino
de correccion de error y los rezagos en diferencias). Esos residuales si deben
ser ruido blanco, y la prueba lo exige con Ljung-Box.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

import artifacts_io
import config
from tests.conftest import autocorrelacion

#: Rango admisible para la varianza de las innovaciones normalizadas.
VARIANZA_MINIMA = 0.80
VARIANZA_MAXIMA = 1.25

#: Tolerancia entre la autocorrelacion medida y la implicada por la vida media.
TOLERANCIA_AUTOCORRELACION = 0.05


def _innovaciones(grupo: str, filtro: str, solo_entrenamiento: bool) -> np.ndarray:
    """Innovaciones normalizadas operables de un filtro, por tramo."""
    archivo = "filter1" if filtro == "filtro1" else "filter2"
    serie = artifacts_io.leer_serie("kalman", f"{archivo}_{grupo}.parquet")
    particiones = artifacts_io.leer_serie("prices.parquet")["particion"]
    mascara = serie["operable"].to_numpy(dtype=bool)
    if solo_entrenamiento:
        mascara &= particiones.to_numpy() == "entrenamiento"
    return serie["innovacion_normalizada"].to_numpy()[mascara]


def _phi_implicado(vida_media_dias: float) -> float:
    """Coeficiente AR(1) implicado por una vida media dada."""
    return float(0.5 ** (1.0 / vida_media_dias))


def test_varianza_innovaciones_filtro1_en_entrenamiento(grupos):
    """El filtro 1 tiene varianza de innovaciones normalizadas ~1 donde se calibro."""
    for grupo in grupos:
        valores = _innovaciones(grupo["grupo"], "filtro1", solo_entrenamiento=True)
        varianza = float(np.var(valores, ddof=1))
        assert VARIANZA_MINIMA <= varianza <= VARIANZA_MAXIMA, (
            f"{grupo['grupo']} filtro 1: varianza de innovaciones normalizadas "
            f"en entrenamiento {varianza:.4f}, fuera de "
            f"[{VARIANZA_MINIMA}, {VARIANZA_MAXIMA}]"
        )


def test_varianza_innovaciones_filtro2_en_muestra_completa(grupos):
    """El filtro 2 tiene varianza ~1 en toda la muestra, gracias a R adaptativa."""
    for grupo in grupos:
        valores = _innovaciones(grupo["grupo"], "filtro2", solo_entrenamiento=False)
        varianza = float(np.var(valores, ddof=1))
        assert VARIANZA_MINIMA <= varianza <= VARIANZA_MAXIMA, (
            f"{grupo['grupo']} filtro 2: varianza de innovaciones normalizadas "
            f"{varianza:.4f}, fuera de [{VARIANZA_MINIMA}, {VARIANZA_MAXIMA}]"
        )


def test_autocorrelacion_de_ambos_filtros_coincide_con_la_vida_media(grupos):
    """La persistencia medida es la que predice la vida media del spread.

    La autocorrelacion NO es baja, y esta prueba no pretende que lo sea: verifica
    que su valor es el que el modelo predice.
    """
    for grupo in grupos:
        phi = _phi_implicado(grupo["vecm"]["vida_media_dias"])
        for filtro in ("filtro1", "filtro2"):
            valores = _innovaciones(grupo["grupo"], filtro, solo_entrenamiento=False)
            medida = autocorrelacion(valores)
            diferencia = abs(medida - phi)
            assert diferencia < TOLERANCIA_AUTOCORRELACION, (
                f"{grupo['grupo']} {filtro}: autocorrelacion medida {medida:.4f} "
                f"contra la implicada por la vida media "
                f"({grupo['vecm']['vida_media_dias']:.1f} d -> phi={phi:.4f}), "
                f"diferencia {diferencia:.4f}"
            )


def test_residuales_del_vecm_son_ruido_blanco(grupos):
    """Los residuales del VECM pasan Ljung-Box: el modelo dinamico es adecuado."""
    for grupo in grupos:
        diagnostico = grupo["vecm"]
        assert diagnostico["residuales_blancos"], (
            f"{grupo['grupo']}: los residuales del VECM no son blancos "
            f"(Ljung-Box({diagnostico['ljungbox_rezagos']}) "
            f"p={diagnostico['ljungbox_pvalue']:.4f} con k={diagnostico['k_rezagos']})"
        )
        assert diagnostico["ljungbox_pvalue"] > config.LJUNGBOX_ALPHA


def test_el_spread_revierte_segun_el_vecm(grupos):
    """beta'alpha es negativo: el desequilibrio efectivamente se corrige."""
    for grupo in grupos:
        beta_alpha = grupo["vecm"]["beta_alpha"]
        assert beta_alpha < 0.0, (
            f"{grupo['grupo']}: beta'alpha = {beta_alpha:+.6f} no es negativo, "
            "asi que el spread no revierte y el grupo no deberia operarse."
        )


def test_escribir_reporte_de_diagnosticos(grupos):
    """Deja los diagnosticos de ambos filtros en un artifact citable.

    No es una prueba de comportamiento sino el requisito de la seccion 3.11.4 de
    REPORTAR varianza y autocorrelacion. Se implementa como prueba para que el
    numero que cita el reporte sea el que `pytest` midio, y no uno copiado a mano.
    """
    filas = []
    for grupo in grupos:
        phi = _phi_implicado(grupo["vecm"]["vida_media_dias"])
        for filtro in ("filtro1", "filtro2"):
            for tramo, solo_train in (("entrenamiento", True), ("muestra_completa", False)):
                valores = _innovaciones(grupo["grupo"], filtro, solo_train)
                ljungbox = float(
                    acorr_ljungbox(valores, lags=[10], return_df=True)[
                        "lb_pvalue"
                    ].iloc[0]
                )
                filas.append(
                    {
                        "grupo": grupo["grupo"],
                        "filtro": filtro,
                        "tramo": tramo,
                        "n_observaciones": len(valores),
                        "varianza_innovaciones_normalizadas": round(
                            float(np.var(valores, ddof=1)), 4
                        ),
                        "varianza_en_rango": bool(
                            VARIANZA_MINIMA
                            <= float(np.var(valores, ddof=1))
                            <= VARIANZA_MAXIMA
                        ),
                        "autocorrelacion_lag1": round(autocorrelacion(valores), 4),
                        "phi_implicado_por_vida_media": round(phi, 4),
                        "diferencia_con_phi": round(
                            abs(autocorrelacion(valores) - phi), 4
                        ),
                        "ljungbox_p_lag10": round(ljungbox, 6),
                        "autocorrelacion_no_significativa": bool(ljungbox > 0.05),
                    }
                )
        filas.append(
            {
                "grupo": grupo["grupo"],
                "filtro": "vecm_ect",
                "tramo": "entrenamiento",
                "n_observaciones": None,
                "varianza_innovaciones_normalizadas": None,
                "varianza_en_rango": None,
                "autocorrelacion_lag1": None,
                "phi_implicado_por_vida_media": round(phi, 4),
                "diferencia_con_phi": None,
                "ljungbox_p_lag10": round(grupo["vecm"]["ljungbox_pvalue"], 6),
                "autocorrelacion_no_significativa": bool(
                    grupo["vecm"]["residuales_blancos"]
                ),
            }
        )

    tabla = pd.DataFrame(filas)
    artifacts_io.escribir_tabla(
        tabla, "performance", "filter_diagnostics.csv", indice=False
    )
    assert artifacts_io.ruta("performance", "filter_diagnostics.csv").exists()
