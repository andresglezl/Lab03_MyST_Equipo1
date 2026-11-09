"""Logica de senales sobre el z-score (seccion 3.6).

Reglas de operacion:

  - Entrada LARGA en el spread si z_t < -z_entrada estando plano. El portafolio
    esta por debajo de su equilibrio: se compra 1 unidad de la pata numeraria y
    se toman las demas patas con el signo que indique beta.
  - Entrada CORTA en el spread si z_t > +z_entrada estando plano.
  - Salida si |z_t| < z_salida: el spread volvio a su equilibrio.
  - Stop si |z_t| > z_stop: el spread se fue MAS en contra. Este stop es el que
    protege contra el modo de perdida real de la estrategia, que no es una
    sucesion de perdidas chicas sino una posicion que deja de revertir.
  - Tenencia maxima: si la posicion lleva abierta mas de
    `MULTIPLO_TENENCIA_MAXIMA` vidas medias, se cierra. Una posicion que no
    revirtio en dos vidas medias ya no es la operacion que creiamos abrir.

Convencion de ejecucion declarada: z_t se calcula con precios de cierre del dia
t y la operacion se ejecuta al cierre del MISMO dia t, como una orden
market-on-close. No hay look-ahead: z_t solo usa informacion hasta t. La
simplificacion que si asumimos es que se consigue el precio de cierre exacto,
sin slippage, lo que se declara como limitacion.

El estado de posicion resultante en t se mantiene para calcular el P&L de t+1,
de modo que el retorno del dia t+1 usa la posicion decidida al cierre de t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

#: Razones por las que se cierra una posicion, para la bitacora de operaciones.
RAZONES_SALIDA = ("reversion", "stop", "tenencia_maxima", "fin_de_muestra")


@dataclass(frozen=True)
class Umbrales:
    """Parametros de la regla de trading.

    Attributes:
        z_entrada: umbral de entrada en valor absoluto del z-score.
        z_salida: umbral de salida en valor absoluto del z-score.
        z_stop: umbral de stop en valor absoluto del z-score.
        tenencia_maxima: dias habiles maximos con la posicion abierta.
    """

    z_entrada: float
    z_salida: float
    z_stop: float
    tenencia_maxima: int

    def como_dict(self) -> dict:
        """Devuelve los umbrales como diccionario serializable."""
        return {
            "z_entrada": self.z_entrada,
            "z_salida": self.z_salida,
            "z_stop": self.z_stop,
            "tenencia_maxima": self.tenencia_maxima,
        }


def tenencia_maxima_desde_vida_media(vida_media_dias: float) -> int:
    """Calibra el periodo maximo de tenencia a la vida media del spread.

    Args:
        vida_media_dias: vida media estimada en entrenamiento.

    Returns:
        Numero entero de dias habiles.
    """
    return int(round(config.MULTIPLO_TENENCIA_MAXIMA * vida_media_dias))


def generar_senales(
    z: pd.Series, operable: pd.Series, umbrales: Umbrales
) -> pd.DataFrame:
    """Recorre el z-score y produce el estado de posicion dia por dia.

    Args:
        z: serie del z-score ex-ante.
        operable: serie booleana que marca los dias posteriores al burn-in.
        umbrales: parametros de la regla.

    Returns:
        DataFrame con 'posicion' (-1, 0, +1), 'dias_en_posicion',
        'evento' ('entrada_larga', 'entrada_corta', 'salida' o vacio) y
        'razon_salida'.
    """
    valores = z.to_numpy(dtype=float)
    puede_operar = operable.to_numpy(dtype=bool)
    n = len(valores)

    posicion = np.zeros(n, dtype=int)
    dias_abierta = np.zeros(n, dtype=int)
    eventos = np.array([""] * n, dtype=object)
    razones = np.array([""] * n, dtype=object)

    actual = 0
    contador = 0

    for t in range(n):
        if not puede_operar[t]:
            posicion[t] = 0
            continue

        z_t = valores[t]

        if actual == 0:
            if z_t < -umbrales.z_entrada:
                actual, contador = 1, 1
                eventos[t] = "entrada_larga"
            elif z_t > umbrales.z_entrada:
                actual, contador = -1, 1
                eventos[t] = "entrada_corta"
        else:
            contador += 1
            if abs(z_t) < umbrales.z_salida:
                razones[t], eventos[t] = "reversion", "salida"
                actual, contador = 0, 0
            elif abs(z_t) > umbrales.z_stop:
                razones[t], eventos[t] = "stop", "salida"
                actual, contador = 0, 0
            elif contador > umbrales.tenencia_maxima:
                razones[t], eventos[t] = "tenencia_maxima", "salida"
                actual, contador = 0, 0

        posicion[t] = actual
        dias_abierta[t] = contador

    if actual != 0:
        razones[n - 1] = "fin_de_muestra"

    return pd.DataFrame(
        {
            "posicion": posicion,
            "dias_en_posicion": dias_abierta,
            "evento": eventos,
            "razon_salida": razones,
        },
        index=z.index,
    )


def pesos_objetivo(
    betas: pd.DataFrame, precios: pd.DataFrame, posicion: pd.Series
) -> pd.DataFrame:
    """Convierte beta y la direccion en pesos de DOLARES por pata.

    El vector beta esta en ACCIONES: dice cuantos titulos de cada activo forman
    una unidad del portafolio reversivo. Para repartir capital hay que pasar a
    dolares multiplicando por el precio, porque una accion de un activo de 350
    dolares no pesa lo mismo que una de un activo de 30:

        w_{j,t} = d_t * beta_{j,t} P_{j,t} / sum_k |beta_{k,t} P_{k,t}|

    La version anterior omitia los precios y repartia el capital como si todos
    los activos costaran lo mismo, lo que describia un portafolio distinto del
    que el motor opera.

    Esta funcion es de REPORTE: deja los pesos en dolares en el artifact de
    senales para el dashboard y el reporte. La contabilidad del backtest se
    lleva en acciones, en `backtest/walkforward.py`.

    Args:
        betas: DataFrame con una columna `beta_{ticker}` por activo.
        precios: panel de precios del grupo.
        posicion: serie de direccion en {-1, 0, +1}.

    Returns:
        DataFrame de pesos con columnas `peso_{ticker}`.
    """
    tickers = [columna.replace("beta_", "") for columna in betas.columns]
    valores = pd.DataFrame(
        betas.to_numpy() * precios[tickers].to_numpy(),
        index=betas.index,
        columns=[f"peso_{ticker}" for ticker in tickers],
    )
    bruto = valores.abs().sum(axis=1)
    return valores.div(bruto, axis=0).mul(posicion, axis=0).fillna(0.0)
