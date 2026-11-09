"""Atribucion de costos: de P&L bruto a P&L neto (seccion 3.8-Dashboard).

La grafica de atribucion de costos que pide el dashboard descompone el P&L bruto
en P&L neto, comisiones totales pagadas y costos totales de prestamo. La
identidad que se reporta y que debe cerrar exactamente es

    P&L bruto = P&L neto + comisiones + prestamo

Se reporta por particion y por grupo, para poder responder dos preguntas
distintas: cuanto del resultado se lo comieron los costos, y cual de los dos
grupos los genero.
"""

from __future__ import annotations

import pandas as pd

import config


def atribucion_de_costos(serie: pd.DataFrame) -> pd.DataFrame:
    """Descompone el P&L bruto en neto, comisiones y prestamo por particion.

    Args:
        serie: serie diaria del portafolio con 'pnl_bruto', 'comision',
            'prestamo', 'pnl_neto' y 'particion'.

    Returns:
        DataFrame con una fila por particion mas una fila 'global'.
    """
    filas = []
    for particion in (*config.NOMBRES_PARTICIONES, "global"):
        tramo = serie if particion == "global" else serie[serie["particion"] == particion]
        if tramo.empty:
            continue
        bruto = float(tramo["pnl_bruto"].sum())
        comision = float(tramo["comision"].sum())
        prestamo = float(tramo["prestamo"].sum())
        neto = float(tramo["pnl_neto"].sum())
        filas.append(
            {
                "particion": particion,
                "pnl_bruto": round(bruto, 2),
                "comisiones": round(comision, 2),
                "prestamo": round(prestamo, 2),
                "pnl_neto": round(neto, 2),
                "residuo_identidad": round(bruto - comision - prestamo - neto, 8),
                # El cociente costos / bruto solo tiene lectura cuando el bruto
                # es POSITIVO: dice que fraccion de la ganancia se fue en
                # costos. Con bruto negativo o cercano a cero el cociente se
                # dispara sin significar nada (un bruto de -2,042 contra 27,386
                # de costos daria "1341%"), asi que se deja vacio y en su lugar
                # se reporta el costo sobre el capital medio del tramo.
                "costos_sobre_bruto": round((comision + prestamo) / bruto, 6)
                if bruto > 0.0
                else None,
                "costos_sobre_capital_medio": round(
                    (comision + prestamo) / float(tramo["capital"].mean()), 6
                ),
            }
        )
    return pd.DataFrame(filas)


def atribucion_por_grupo(
    pnl_bruto: pd.DataFrame,
    comision: pd.DataFrame,
    prestamo: pd.DataFrame,
    etiquetas: pd.Series,
) -> pd.DataFrame:
    """Descompone el P&L por grupo y particion.

    Args:
        pnl_bruto: P&L bruto diario por grupo, en dolares.
        comision: comision diaria por grupo.
        prestamo: prestamo diario por grupo.
        etiquetas: serie de particiones por fecha.

    Returns:
        DataFrame con una fila por grupo y particion.
    """
    filas = []
    for grupo in pnl_bruto.columns:
        for particion in (*config.NOMBRES_PARTICIONES, "global"):
            mascara = (
                pd.Series(True, index=pnl_bruto.index)
                if particion == "global"
                else pd.Series(etiquetas.values == particion, index=pnl_bruto.index)
            )
            bruto = float(pnl_bruto.loc[mascara, grupo].sum())
            com = float(comision.loc[mascara, grupo].sum())
            pres = float(prestamo.loc[mascara, grupo].sum())
            filas.append(
                {
                    "grupo": grupo,
                    "particion": particion,
                    "pnl_bruto": round(bruto, 2),
                    "comisiones": round(com, 2),
                    "prestamo": round(pres, 2),
                    "pnl_neto": round(bruto - com - pres, 2),
                }
            )
    return pd.DataFrame(filas)
