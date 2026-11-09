"""VECM con el vector cointegrante de Johansen como termino de correccion de error.

Seccion 3.6: "Estime un modelo VECM usando el vector cointegrante de Johansen
como termino de correccion de error".

Decision declarada: el vector beta NO se reestima. Se fija al vector normalizado
que salio de Johansen en entrenamiento y se estima por OLS, ecuacion por
ecuacion, el resto del sistema:

    Delta P_{i,t} = a_i + alpha_i ECT_{t-1} + sum_{l=1..k} Gamma_{i,l} Delta P_{t-l} + e_{i,t}
    ECT_t = beta' P_t

Estimar asi, con beta fijo, es la segunda etapa del procedimiento de dos pasos y
tiene dos ventajas sobre dejar que `statsmodels.VECM` reestime su propio beta: el
termino de correccion de error que entra al VECM es EXACTAMENTE el mismo spread
que operamos, y los alpha son directamente interpretables como la velocidad con
que cada pata corrige el desequilibrio que medimos.

Velocidad de reversion del spread
---------------------------------
La cantidad relevante NO es el alpha de una sola pata, sino la combinacion

    beta' alpha

que es la velocidad de ajuste del propio termino de correccion de error:
premultiplicando el sistema por beta' se obtiene

    Delta ECT_t = (beta' alpha) ECT_{t-1} + (terminos en diferencias) + beta' e_t

de donde la vida media del spread es h = -ln(2) / ln(1 + beta'alpha). Una pata
individual puede tener alpha positivo (desestabilizador) y el spread revertir de
todas formas, porque lo que importa es el efecto neto ponderado por beta. Este
modulo reporta las dos cosas: beta'alpha como medida de reversion y los alpha
individuales como atribucion de quien hace la correccion.

Seleccion del orden de rezagos
------------------------------
Se elige el MENOR k de `config.MALLA_REZAGOS_VECM` cuyos residuales de la forma
reducida del ECT pasan Ljung-Box al 5%, evaluado en k + 10 rezagos, es decir mas
alla de los rezagos ajustados. Evaluarlo dentro de los rezagos ajustados seria
circular: la regresion absorbe por construccion esas autocorrelaciones y el
p-value sale artificialmente cerca de 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

import config


def _matriz_rezagos(diferencias: pd.DataFrame, k_rezagos: int) -> pd.DataFrame:
    """Construye las columnas de diferencias rezagadas del VECM."""
    bloques = {}
    for rezago in range(1, k_rezagos + 1):
        desplazado = diferencias.shift(rezago)
        for columna in diferencias.columns:
            bloques[f"d{columna}_lag{rezago}"] = desplazado[columna]
    return pd.DataFrame(bloques, index=diferencias.index)


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Resuelve OLS y devuelve coeficientes, residuales y varianza residual."""
    coeficientes, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuo = y - X @ coeficientes
    grados_libertad = len(y) - X.shape[1]
    return coeficientes, residuo, float(residuo @ residuo / grados_libertad)


def calcular_ect(precios: pd.DataFrame, tickers: list[str], beta: np.ndarray) -> pd.Series:
    """Calcula el termino de correccion de error ECT_t = beta' P_t.

    Args:
        precios: panel de precios del grupo.
        tickers: activos en el orden del vector beta.
        beta: vector cointegrante normalizado.

    Returns:
        Serie del ECT indexada por fecha.
    """
    return pd.Series(precios[tickers].to_numpy() @ beta, index=precios.index, name="ect")


def dinamica_reducida_ect(ect: pd.Series, k_rezagos: int) -> dict:
    """Estima la forma reducida univariada del ECT y prueba sus residuales.

    Ajusta Delta ECT_t = c + phi ECT_{t-1} + sum_l gamma_l Delta ECT_{t-l} + eps_t.

    Args:
        ect: serie del termino de correccion de error.
        k_rezagos: numero de rezagos en diferencias.

    Returns:
        Diccionario con 'phi', 'vida_media_dias', los residuales, el p-value de
        Ljung-Box y el numero de rezagos en que se evaluo.
    """
    diferencia = ect.diff()
    columnas = {"ect_lag1": ect.shift(1)}
    for rezago in range(1, k_rezagos + 1):
        columnas[f"dect_lag{rezago}"] = diferencia.shift(rezago)
    diseno = pd.DataFrame(columnas, index=ect.index)
    diseno["constante"] = 1.0

    muestra = pd.concat([diferencia.rename("objetivo"), diseno], axis=1).dropna()
    coeficientes, residuo, _ = _ols(
        muestra[diseno.columns].to_numpy(), muestra["objetivo"].to_numpy()
    )

    phi = float(coeficientes[0])
    rezagos_prueba = k_rezagos + config.LJUNGBOX_REZAGOS_EXTRA
    pvalue = float(
        acorr_ljungbox(residuo, lags=[rezagos_prueba], return_df=True)[
            "lb_pvalue"
        ].iloc[0]
    )

    return {
        "phi": phi,
        "vida_media_dias": float(-np.log(2.0) / np.log(1.0 + phi))
        if -1.0 < phi < 0.0
        else float("nan"),
        "residuales": pd.Series(residuo, index=muestra.index, name="residual_ect"),
        "ljungbox_pvalue": pvalue,
        "ljungbox_rezagos": rezagos_prueba,
        "residuales_blancos": pvalue > config.LJUNGBOX_ALPHA,
        "k_rezagos": k_rezagos,
    }


def seleccionar_rezagos(
    ect: pd.Series, malla: tuple[int, ...] = config.MALLA_REZAGOS_VECM
) -> tuple[int, pd.DataFrame]:
    """Elige el menor orden de rezagos que blanquea los residuales del ECT.

    Args:
        ect: serie del ECT en entrenamiento.
        malla: ordenes de rezago a probar, en orden creciente.

    Returns:
        Tupla `(k_elegido, tabla)`. Si ningun k de la malla blanquea los
        residuales se devuelve el de mayor p-value y la tabla lo deja ver.
    """
    filas = []
    for k in malla:
        resultado = dinamica_reducida_ect(ect, k)
        filas.append(
            {
                "k_rezagos": k,
                "phi": round(resultado["phi"], 6),
                "vida_media_dias": round(resultado["vida_media_dias"], 2),
                "ljungbox_rezagos": resultado["ljungbox_rezagos"],
                "ljungbox_pvalue": round(resultado["ljungbox_pvalue"], 6),
                "residuales_blancos": resultado["residuales_blancos"],
            }
        )
    tabla = pd.DataFrame(filas)

    blancos = tabla[tabla["residuales_blancos"]]
    k_elegido = int(
        blancos["k_rezagos"].iloc[0]
        if not blancos.empty
        else tabla.loc[tabla["ljungbox_pvalue"].idxmax(), "k_rezagos"]
    )
    tabla["elegido"] = tabla["k_rezagos"] == k_elegido
    return k_elegido, tabla


def estimar_vecm(
    precios: pd.DataFrame,
    tickers: list[str],
    beta: np.ndarray,
    k_rezagos: int,
) -> dict:
    """Estima el VECM por OLS con el ECT fijo al vector de Johansen.

    Args:
        precios: panel del grupo restringido a la particion de entrenamiento.
        tickers: activos del grupo en el orden del vector beta.
        beta: vector cointegrante normalizado.
        k_rezagos: numero de rezagos en diferencias.

    Returns:
        Diccionario con los coeficientes por ecuacion, los residuales, beta'alpha,
        la vida media implicita y la dinamica reducida del ECT.
    """
    niveles = precios[tickers]
    ect = calcular_ect(precios, tickers, beta)
    diferencias = niveles.diff()

    diseno = pd.concat(
        [ect.shift(1).rename("ect_lag1"), _matriz_rezagos(diferencias, k_rezagos)],
        axis=1,
    )
    diseno["constante"] = 1.0
    muestra = pd.concat([diferencias, diseno], axis=1).dropna()

    X = muestra[diseno.columns].to_numpy()
    inversa = np.linalg.inv(X.T @ X)

    ecuaciones = {}
    residuales = pd.DataFrame(index=muestra.index)
    for ticker in tickers:
        coeficientes, residuo, varianza = _ols(X, muestra[ticker].to_numpy())
        errores_estandar = np.sqrt(varianza * np.diag(inversa))
        residuales[ticker] = residuo
        ecuaciones[ticker] = {
            "alpha": float(coeficientes[0]),
            "alpha_error_estandar": float(errores_estandar[0]),
            "alpha_t": float(coeficientes[0] / errores_estandar[0]),
            "varianza_residual": varianza,
        }

    alphas = np.array([ecuaciones[ticker]["alpha"] for ticker in tickers])
    beta_alpha = float(beta @ alphas)
    vida_media = (
        float(-np.log(2.0) / np.log(1.0 + beta_alpha))
        if -1.0 < beta_alpha < 0.0
        else float("nan")
    )
    reducida = dinamica_reducida_ect(ect, k_rezagos)

    return {
        "ecuaciones": ecuaciones,
        "residuales": residuales,
        "ect": ect,
        "alphas": alphas,
        "beta_alpha": beta_alpha,
        "vida_media_dias": vida_media,
        "reducida_ect": reducida,
        "k_rezagos": k_rezagos,
        "n_observaciones": int(len(muestra)),
        "revierte": beta_alpha < 0.0,
    }


def tabla_alphas(resultado: dict, tickers: list[str]) -> pd.DataFrame:
    """Resume los coeficientes de correccion de error de todas las patas.

    Args:
        resultado: salida de `estimar_vecm`.
        tickers: activos del grupo en el orden del vector beta.

    Returns:
        DataFrame con alpha, su error estandar, su t y si corrige o desestabiliza.
    """
    filas = [
        {
            "ticker": ticker,
            "alpha": round(resultado["ecuaciones"][ticker]["alpha"], 6),
            "error_estandar": round(
                resultado["ecuaciones"][ticker]["alpha_error_estandar"], 6
            ),
            "t": round(resultado["ecuaciones"][ticker]["alpha_t"], 4),
            "significativo_5pct": abs(resultado["ecuaciones"][ticker]["alpha_t"]) > 1.96,
            "corrige_desequilibrio": resultado["ecuaciones"][ticker]["alpha"] < 0,
        }
        for ticker in tickers
    ]
    return pd.DataFrame(filas)
