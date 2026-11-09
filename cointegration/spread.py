"""Paso 5 del embudo: normalizacion del vector y validacion del spread (3.2.5).

Contiene la convencion de normalizacion y signo (C2 del plan de fases), que es
la decision de diseno mas consultada en la exposicion:

    i* = argmax_i |beta_i^J|,     beta <- beta^J / beta^J_{i*}

Dividir por la componente CON SU SIGNO (no por su valor absoluto) resuelve de un
golpe las dos indeterminaciones del eigenvector de Johansen: la de escala y la
de signo, dejando beta_{i*} = +1 exactamente. Consecuencias:

  - La pata numeraria i* se elige UNA sola vez sobre la particion de
    entrenamiento y queda congelada. Re-elegirla por ventana romperia la prueba
    obligatoria 2 (estabilidad de escala y signo entre reestimaciones).
  - "Largo el spread" significa siempre largo 1 unidad de la pata numeraria, de
    modo que el sentido de la senal no se invierte a lo largo de la muestra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

import config


def elegir_pata_numeraria(vector: np.ndarray, tickers: list[str]) -> tuple[int, str]:
    """Elige la pata de normalizacion como la de mayor peso absoluto.

    Args:
        vector: columna cointegrante sin normalizar.
        tickers: nombres de los activos en el mismo orden que `vector`.

    Returns:
        Tupla `(indice, ticker)` de la pata numeraria.
    """
    indice = int(np.argmax(np.abs(vector)))
    return indice, tickers[indice]


def normalizar_vector(vector: np.ndarray, indice_numerario: int) -> np.ndarray:
    """Normaliza el vector cointegrante dividiendo por la pata numeraria.

    Args:
        vector: columna cointegrante sin normalizar.
        indice_numerario: posicion de la pata numeraria.

    Returns:
        Vector con la componente numeraria igual a +1 exactamente.

    Raises:
        ValueError: si la componente numeraria es cero y la escala no se puede
            fijar.
    """
    pivote = vector[indice_numerario]
    if np.isclose(pivote, 0.0):
        raise ValueError(
            "La componente de la pata numeraria es cero: el vector no se puede "
            "normalizar sobre esa pata."
        )
    return vector / pivote


def construir_spread(precios: pd.DataFrame, beta: np.ndarray, tickers: list[str]) -> pd.Series:
    """Construye el spread del portafolio S_t = beta' P_t.

    Args:
        precios: panel de precios.
        beta: vector de pesos normalizado.
        tickers: activos en el mismo orden que `beta`.

    Returns:
        Serie del spread indexada por fecha.
    """
    return pd.Series(
        precios[tickers].to_numpy() @ beta, index=precios.index, name="spread"
    )


def prueba_adf(serie: pd.Series) -> dict:
    """Aplica ADF con constante a una serie y devuelve estadistico y p-value.

    Se usa `regression='c'` porque el spread construido con `det_order=0` tiene
    un nivel de equilibrio distinto de cero, no una media forzada a cero.

    Args:
        serie: serie a probar (tipicamente el spread).

    Returns:
        Diccionario con 'adf_estadistico', 'adf_pvalue' y 'adf_rezagos'.
    """
    estadistico, pvalue, rezagos, *_ = adfuller(
        serie.dropna().to_numpy(), regression="c", autolag="aic"
    )
    return {
        "adf_estadistico": float(estadistico),
        "adf_pvalue": float(pvalue),
        "adf_rezagos": int(rezagos),
    }


def vida_media(serie: pd.Series) -> float:
    """Estima la vida media de reversion ajustando un proceso de Ornstein-Uhlenbeck.

    Se estima por OLS la version discreta

        Delta S_t = a + b * S_{t-1} + e_t

    y la vida media es h = -ln(2) / ln(1 + b). Solo esta definida si
    -1 < b < 0, es decir si la serie efectivamente revierte.

    Args:
        serie: serie del spread.

    Returns:
        Vida media en dias habiles, o `np.nan` si la serie no revierte.
    """
    valores = serie.dropna().to_numpy()
    rezago = valores[:-1]
    diferencia = np.diff(valores)
    pendiente = np.polyfit(rezago, diferencia, 1)[0]
    if not -1.0 < pendiente < 0.0:
        return float("nan")
    return float(-np.log(2.0) / np.log(1.0 + pendiente))


def evaluar_columna_cointegrante(
    precios: pd.DataFrame, vector: np.ndarray, tickers: list[str]
) -> dict:
    """Normaliza una columna cointegrante y mide la reversion de su spread.

    Args:
        precios: panel de entrenamiento.
        vector: columna cointegrante sin normalizar.
        tickers: activos en el mismo orden que `vector`.

    Returns:
        Diccionario con la pata numeraria, el vector normalizado, el resultado
        del ADF, la vida media y si el spread es operable.
    """
    indice, numerario = elegir_pata_numeraria(vector, tickers)
    beta = normalizar_vector(vector, indice)
    spread = construir_spread(precios, beta, tickers)

    resultado = {
        "pata_numeraria": numerario,
        "indice_numerario": indice,
        "beta": beta,
        "beta_por_ticker": dict(zip(tickers, beta.round(6))),
        **prueba_adf(spread),
        "vida_media_dias": vida_media(spread),
    }
    resultado["pasa_adf_spread"] = resultado["adf_pvalue"] < config.PVALUE_MAXIMO
    resultado["vida_media_operable"] = bool(
        config.VIDA_MEDIA_MINIMA
        <= (resultado["vida_media_dias"] if resultado["vida_media_dias"] == resultado["vida_media_dias"] else -1)
        <= config.VIDA_MEDIA_MAXIMA
    )
    return resultado


def seleccionar_columna_mas_reversiva(
    precios: pd.DataFrame, resultado_johansen: dict
) -> tuple[dict, pd.DataFrame]:
    """Elige entre las columnas cointegrantes la de spread mas estacionario.

    La seccion 3.2 indica que un eigenvalor mayor corresponde a una correccion
    de error mas fuerte, pero tambien que cuando hay mas de una relacion de
    cointegracion se debe construir el spread de cada columna, evaluar ADF y
    vida media, y seleccionar empiricamente. Eso es lo que se hace aqui:
    criterio primario el p-value del ADF (mas bajo gana), desempate por vida
    media mas corta.

    Si el rango estimado es 0 se evalua de todas formas la primera columna, para
    poder reportar su ADF y su vida media en la tabla de grupos probados.

    Args:
        precios: panel de entrenamiento.
        resultado_johansen: salida de `cointegration.johansen.prueba_johansen`.

    Returns:
        Tupla `(mejor_columna, tabla_comparativa)`.
    """
    tickers = resultado_johansen["tickers"]
    eigenvectores = resultado_johansen["eigenvectores"]
    n_columnas = max(resultado_johansen["rango"], 1)

    evaluaciones = []
    for columna in range(n_columnas):
        evaluacion = evaluar_columna_cointegrante(
            precios, eigenvectores[:, columna], tickers
        )
        evaluacion["columna"] = columna
        evaluacion["eigenvalor"] = float(resultado_johansen["eigenvalores"][columna])
        evaluaciones.append(evaluacion)

    def clave_orden(item: dict) -> tuple[float, float]:
        vida = item["vida_media_dias"]
        return (item["adf_pvalue"], vida if vida == vida else float("inf"))

    mejor = min(evaluaciones, key=clave_orden)

    tabla = pd.DataFrame(
        [
            {
                "columna": e["columna"],
                "eigenvalor": round(e["eigenvalor"], 8),
                "pata_numeraria": e["pata_numeraria"],
                "adf_estadistico": round(e["adf_estadistico"], 4),
                "adf_pvalue": round(e["adf_pvalue"], 6),
                "vida_media_dias": round(e["vida_media_dias"], 2)
                if e["vida_media_dias"] == e["vida_media_dias"]
                else None,
                "seleccionada": e["columna"] == mejor["columna"],
                **{f"beta_{t}": round(v, 6) for t, v in e["beta_por_ticker"].items()},
            }
            for e in evaluaciones
        ]
    )
    return mejor, tabla
