"""Dimensionamiento de posiciones en acciones y dolares (seccion 3.1 y C6).

El limite del 80% aplica al capital del portafolio CONSOLIDADO y se interpreta
como nocional BRUTO, es decir la suma de los valores absolutos de todas las
patas de todos los grupos. Es la lectura conservadora: una posicion de spread
compra una canasta y vende otra, y el bruto es el capital que el broker
realmente necesita ver para sostener las dos.

Con C_{t-1} el capital consolidado al cierre del dia anterior y w_g el peso de
Risk Parity del grupo g, el nocional asignado al grupo es

    N_{g,t} = EXPOSICION_MAXIMA * C_{t-1} * w_g

y el numero de unidades del portafolio reversivo que ese nocional compra es

    u_{g,t} = N_{g,t} / sum_k |beta_hat_{k,t}| P_{k,t}

de donde las acciones de cada pata son

    q_{j,t} = d_{g,t} * u_{g,t} * beta_hat_{j,t},     d in {-1, 0, +1}

Se usa C_{t-1} y no C_t para evitar la circularidad de dimensionar con un
capital que todavia depende del resultado del dia. Las acciones se dejan
fraccionarias: redondearlas a enteros con un capital de un millon introduciria
un error de redondeo irrelevante frente al tamano de las posiciones y
complicaria la conciliacion de la prueba obligatoria 3.
"""

from __future__ import annotations

import numpy as np

import config


def unidades_del_portafolio(
    betas: np.ndarray, precios: np.ndarray, nocional_asignado: float
) -> float:
    """Unidades del portafolio reversivo que caben en el nocional asignado.

    Args:
        betas: vector beta_hat del dia.
        precios: vector de precios del dia.
        nocional_asignado: nocional bruto en dolares asignado al grupo.

    Returns:
        Numero de unidades, o 0.0 si el nocional bruto unitario es cero.
    """
    nocional_unitario = float(np.abs(betas) @ precios)
    if nocional_unitario <= 0.0:
        return 0.0
    return nocional_asignado / nocional_unitario


def acciones_objetivo(
    betas: np.ndarray,
    precios: np.ndarray,
    direccion: int,
    capital_consolidado: float,
    peso_grupo: float,
) -> np.ndarray:
    """Acciones objetivo de cada pata del grupo.

    Args:
        betas: vector beta_hat del dia.
        precios: vector de precios del dia.
        direccion: +1 largo el spread, -1 corto, 0 plano.
        capital_consolidado: capital del portafolio consolidado al cierre previo.
        peso_grupo: peso de Risk Parity del grupo.

    Returns:
        Vector de acciones objetivo, con signo.
    """
    if direccion == 0:
        return np.zeros_like(betas)
    nocional = config.EXPOSICION_MAXIMA * capital_consolidado * peso_grupo
    unidades = unidades_del_portafolio(betas, precios, nocional)
    return direccion * unidades * betas
