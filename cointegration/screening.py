"""Paso 2 del embudo: filtro por correlacion movil (seccion 3.2.2).

Decision declarada: la correlacion se calcula sobre **niveles de precio**, no
sobre retornos. El objetivo del filtro es detectar una tendencia estocastica
comun, que es una propiedad de los niveles; dos activos pueden tener retornos
poco correlacionados y compartir tendencia, o retornos muy correlacionados sin
compartirla. Se reporta ademas la correlacion de retornos como columna
secundaria para que el lector pueda juzgar las dos.

Es un filtro HEURISTICO: la correlacion de niveles entre series con tendencia
esta sesgada al alza (correlacion espuria). Por eso no decide nada por si sola;
el criterio definitivo es Johansen. Su unica funcion es reducir el numero de
grupos que llegan a las pruebas formales, lo que a su vez reduce `m`.
"""

from __future__ import annotations

import itertools

import pandas as pd

import config
from data.universe import ACTIVOS_POR_GRUPO, PoolCandidatos


def generar_grupos_candidatos(pool: PoolCandidatos) -> list[tuple[str, ...]]:
    """Enumera todos los subconjuntos del pool con el tamano de grupo exigido.

    Args:
        pool: pool pre-registrado del universo.

    Returns:
        Lista de tuplas de tickers, ordenada de forma determinista para que la
        tabla de grupos probados sea identica entre corridas.
    """
    return sorted(itertools.combinations(pool.tickers, ACTIVOS_POR_GRUPO))


def correlacion_movil_minima(
    precios: pd.DataFrame,
    grupo: tuple[str, ...],
    ventana: int = config.VENTANA_CORRELACION_MOVIL,
) -> dict[str, float]:
    """Resume la correlacion movil de un grupo por su par mas debil.

    Para cada par del grupo se calcula la correlacion movil de `ventana` dias y
    se toma su MEDIANA en el tiempo (robusta a ventanas atipicas como 2020).
    El grupo se resume por el MINIMO de esas medianas: el par mas debil manda,
    porque un grupo con un activo desacoplado no comparte una sola tendencia.

    Args:
        precios: panel de precios restringido a la particion de entrenamiento.
        grupo: tickers del grupo candidato.
        ventana: tamano de la ventana movil en dias habiles.

    Returns:
        Diccionario con 'corr_niveles_min', 'corr_retornos_min' y el par que
        produjo el minimo de niveles.
    """
    niveles = precios[list(grupo)]
    retornos = niveles.pct_change()

    medianas_niveles: dict[tuple[str, str], float] = {}
    medianas_retornos: dict[tuple[str, str], float] = {}
    for activo_a, activo_b in itertools.combinations(grupo, 2):
        medianas_niveles[(activo_a, activo_b)] = (
            niveles[activo_a].rolling(ventana).corr(niveles[activo_b]).median()
        )
        medianas_retornos[(activo_a, activo_b)] = (
            retornos[activo_a].rolling(ventana).corr(retornos[activo_b]).median()
        )

    par_debil = min(medianas_niveles, key=medianas_niveles.get)
    return {
        "corr_niveles_min": float(medianas_niveles[par_debil]),
        "corr_retornos_min": float(min(medianas_retornos.values())),
        "par_mas_debil": f"{par_debil[0]}/{par_debil[1]}",
    }


def aplicar_filtro_correlacion(
    precios: pd.DataFrame,
    grupo: tuple[str, ...],
    umbral: float = config.CORRELACION_MINIMA,
) -> dict:
    """Evalua si un grupo supera el filtro de correlacion.

    Args:
        precios: panel de entrenamiento.
        grupo: tickers del grupo candidato.
        umbral: correlacion minima exigida al par mas debil.

    Returns:
        El resumen de correlacion mas la llave booleana 'pasa_correlacion'.
    """
    resumen = correlacion_movil_minima(precios, grupo)
    resumen["pasa_correlacion"] = resumen["corr_niveles_min"] >= umbral
    return resumen
