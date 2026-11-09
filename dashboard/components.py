"""Piezas visuales reutilizables del dashboard.

Centraliza el sombreado de particiones y la paleta para que las dos vistas se
vean consistentes y para no repetir el mismo bloque de codigo en cada grafica.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import config

#: Colores por particion, los mismos que usan las figuras del reporte.
COLOR_PARTICION = {
    "entrenamiento": "#1b7837",
    "prueba": "#d9a300",
    "validacion": "#b2182b",
}

#: Paleta para series de activos y grupos.
PALETA = [
    "#2166ac", "#b2182b", "#1b7837", "#8e44ad",
    "#d9a300", "#17a2b8", "#e8702a", "#555555",
]


def rangos_de_particion(particiones: pd.Series) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Primer y ultimo dia de cada particion.

    Args:
        particiones: serie de etiquetas de particion indexada por fecha.

    Returns:
        Lista de tuplas `(nombre, inicio, fin)` en orden cronologico.
    """
    rangos = []
    for nombre in config.NOMBRES_PARTICIONES:
        fechas = particiones.index[particiones.values == nombre]
        if len(fechas):
            rangos.append((nombre, fechas[0], fechas[-1]))
    return rangos


def sombrear_particiones(figura: go.Figure, particiones: pd.Series) -> go.Figure:
    """Sombrea el fondo de la figura segun la particion y la etiqueta.

    Args:
        figura: figura de Plotly a modificar.
        particiones: serie de etiquetas de particion indexada por fecha.

    Returns:
        La misma figura, con las bandas agregadas.
    """
    for nombre, inicio, fin in rangos_de_particion(particiones):
        figura.add_vrect(
            x0=inicio,
            x1=fin,
            fillcolor=COLOR_PARTICION[nombre],
            opacity=0.07,
            layer="below",
            line_width=0,
            annotation_text=nombre,
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color=COLOR_PARTICION[nombre],
        )
    return figura


def aplicar_formato(
    figura: go.Figure, titulo: str, titulo_y: str, altura: int = 420
) -> go.Figure:
    """Aplica titulo, etiquetas de eje y layout uniforme.

    Args:
        figura: figura de Plotly.
        titulo: titulo de la grafica.
        titulo_y: etiqueta del eje vertical.
        altura: altura en pixeles.

    Returns:
        La misma figura, formateada.
    """
    figura.update_layout(
        title=titulo,
        yaxis_title=titulo_y,
        xaxis_title="Fecha",
        height=altura,
        hovermode="x unified",
        margin=dict(l=60, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_white",
    )
    return figura


def formatear_metrica(valor: float, formato: str) -> str:
    """Formatea una metrica para una tarjeta, tolerando valores infinitos.

    Args:
        valor: numero a formatear.
        formato: 'ratio', 'porcentaje' o 'dolares'.

    Returns:
        Texto listo para mostrar.
    """
    if valor is None or not pd.notna(valor):
        return "n/d"
    if valor == float("inf"):
        return "inf"
    if formato == "porcentaje":
        return f"{valor:.2%}"
    if formato == "dolares":
        return f"${valor:,.0f}"
    return f"{valor:.3f}"
