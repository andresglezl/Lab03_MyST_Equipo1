"""Vista ejecutiva del dashboard (seccion 3.8).

Disenada para un portfolio manager o un comite de inversion. Los cinco
elementos que exige el PDF:

1. Curva de capital a lo largo de entrenamiento, prueba y validacion, con el
   drawdown superpuesto.
2. Tarjeta de desempeno: Sharpe, Sortino, Calmar, Maximum Drawdown, Win Rate y
   Profit Factor.
3. Grafica de atribucion de costos: P&L bruto descompuesto en P&L neto,
   comisiones totales y costos totales de prestamo.
4. Tabla de bitacora de operaciones, filtrable por periodo, direccion y
   resultado.
5. Distribucion de retornos: histograma de retornos por operacion con una
   distribucion normal superpuesta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import config
from dashboard import components, data_access


def _curva_de_capital() -> None:
    """Elemento 1: curva de capital con drawdown superpuesto."""
    st.subheader("Curva de capital y drawdown")
    consolidado = data_access.capital_consolidado()
    iguales = data_access.capital_pesos_iguales()
    grupos = [g["grupo"] for g in data_access.grupos_operados()]

    comparar = st.multiselect(
        "Portafolios a comparar",
        options=["Pesos iguales"] + [f"{g} solo" for g in grupos],
        default=["Pesos iguales"],
        help="El portafolio principal (Risk Parity) siempre se muestra.",
    )

    figura = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
        vertical_spacing=0.07,
        subplot_titles=("Capital", "Drawdown del consolidado"),
    )
    figura.add_trace(
        go.Scatter(
            x=consolidado.index, y=consolidado["capital"],
            name="Consolidado Risk Parity",
            line=dict(color=components.PALETA[0], width=2.2),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    if "Pesos iguales" in comparar:
        figura.add_trace(
            go.Scatter(
                x=iguales.index, y=iguales["capital"], name="Pesos iguales",
                line=dict(color=components.PALETA[3], width=1.4, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )
    for posicion, grupo in enumerate(grupos):
        if f"{grupo} solo" not in comparar:
            continue
        serie = data_access.capital_grupo(grupo)
        figura.add_trace(
            go.Scatter(
                x=serie.index, y=serie["capital"], name=f"{grupo} solo",
                line=dict(color=components.PALETA[4 + posicion], width=1.2),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
            ),
            row=1, col=1,
        )
    figura.add_hline(
        y=config.CAPITAL_INICIAL, line=dict(color="black", width=1, dash="dot"),
        row=1, col=1,
    )
    figura.add_trace(
        go.Scatter(
            x=consolidado.index, y=consolidado["drawdown"] * 100,
            name="Drawdown", fill="tozeroy",
            line=dict(color="#b2182b", width=1),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>",
        ),
        row=2, col=1,
    )
    components.sombrear_particiones(figura, consolidado["particion"])
    figura.update_yaxes(title_text="Capital (USD)", row=1, col=1)
    figura.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    figura.update_xaxes(title_text="Fecha", row=2, col=1)
    figura.update_layout(
        height=620, hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
        margin=dict(l=70, r=30, t=70, b=50),
    )
    st.plotly_chart(figura, use_container_width=True)


def _tarjeta_de_desempeno() -> None:
    """Elemento 2: las seis metricas de la tarjeta de desempeno."""
    st.subheader("Tarjeta de desempeno")
    metricas = data_access.metricas()
    portafolios = {
        "Consolidado Risk Parity": "consolidado_risk_parity",
        "Consolidado pesos iguales": "consolidado_pesos_iguales",
        **{
            f"{g['grupo']} solo": f"grupo_{g['grupo']}"
            for g in data_access.grupos_operados()
        },
    }

    columna_izq, columna_der = st.columns(2)
    with columna_izq:
        etiqueta = st.selectbox("Portafolio", list(portafolios))
    with columna_der:
        particion = st.selectbox(
            "Periodo", ["global", *config.NOMBRES_PARTICIONES]
        )

    tarjeta = metricas[portafolios[etiqueta]][particion]
    celdas = st.columns(6)
    valores = [
        ("Sharpe", tarjeta["sharpe"], "ratio"),
        ("Sortino", tarjeta["sortino"], "ratio"),
        ("Calmar", tarjeta["calmar"], "ratio"),
        ("Max Drawdown", tarjeta["max_drawdown"], "porcentaje"),
        ("Win Rate", tarjeta["win_rate"], "porcentaje"),
        ("Profit Factor", tarjeta["profit_factor"], "ratio"),
    ]
    for celda, (nombre, valor, formato) in zip(celdas, valores):
        celda.metric(nombre, components.formatear_metrica(valor, formato))

    detalle = st.columns(4)
    detalle[0].metric(
        "Capital final",
        components.formatear_metrica(tarjeta["capital_final"], "dolares"),
    )
    detalle[1].metric(
        "Retorno del periodo",
        components.formatear_metrica(tarjeta["retorno_total"], "porcentaje"),
    )
    detalle[2].metric(
        "Volatilidad anual",
        components.formatear_metrica(tarjeta["volatilidad_anual"], "porcentaje"),
    )
    detalle[3].metric("Operaciones", f"{tarjeta['n_operaciones']}")

    if particion != "entrenamiento" and tarjeta["sharpe"] < 0:
        st.warning(
            "Sharpe negativo fuera de muestra. La estrategia funciona en "
            "entrenamiento y pierde en prueba y validacion. Vea la seccion de "
            "limitaciones del reporte: con m = 26 grupos probados y 1.3 falsos "
            "positivos esperados, no se puede afirmar que las relaciones de "
            "equilibrio encontradas sean estructurales."
        )


def _atribucion_de_costos() -> None:
    """Elemento 3: P&L bruto descompuesto en neto, comisiones y prestamo."""
    st.subheader("Atribucion de costos")
    tabla = data_access.atribucion_costos()
    orden = [*config.NOMBRES_PARTICIONES, "global"]
    tabla = tabla.set_index("particion").reindex(orden).reset_index()

    figura = go.Figure()
    for nombre, columna, color in (
        ("P&L neto", "pnl_neto", components.PALETA[2]),
        ("Comisiones", "comisiones", components.PALETA[1]),
        ("Prestamo", "prestamo", components.PALETA[4]),
    ):
        figura.add_trace(
            go.Bar(
                x=tabla["particion"], y=tabla[columna], name=nombre,
                marker_color=color,
                hovertemplate="%{x}<br>" + nombre + ": $%{y:,.0f}<extra></extra>",
            )
        )
    figura.add_trace(
        go.Scatter(
            x=tabla["particion"], y=tabla["pnl_bruto"], name="P&L bruto",
            mode="markers+lines", marker=dict(size=11, symbol="diamond",
                                              color="black"),
            line=dict(color="black", width=1, dash="dot"),
            hovertemplate="%{x}<br>Bruto: $%{y:,.0f}<extra></extra>",
        )
    )
    figura.update_layout(
        barmode="relative", title="P&L bruto = P&L neto + comisiones + prestamo",
        yaxis_title="USD", xaxis_title="Particion", height=430,
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
    )
    st.plotly_chart(figura, use_container_width=True)

    global_ = tabla[tabla["particion"] == "global"].iloc[0]
    st.caption(
        f"Los costos son el {global_['costos_sobre_bruto']:.1%} del P&L bruto "
        f"global. La comision (${global_['comisiones']:,.0f}) domina al prestamo "
        f"(${global_['prestamo']:,.0f}): el costo que manda es el 0.125% por "
        "pata con rebalanceo diario de cuatro patas, no el 0.25% anual del corto."
    )


def _bitacora_de_operaciones() -> pd.DataFrame:
    """Elemento 4: bitacora filtrable por periodo, direccion y resultado.

    Returns:
        La bitacora ya filtrada, para reutilizarla en el histograma.
    """
    st.subheader("Bitacora de operaciones")
    operaciones = data_access.operaciones()

    fila = st.columns(4)
    grupos = fila[0].multiselect(
        "Grupo", sorted(operaciones["grupo"].unique()),
        default=sorted(operaciones["grupo"].unique()),
    )
    particiones = fila[1].multiselect(
        "Periodo", list(config.NOMBRES_PARTICIONES),
        default=list(config.NOMBRES_PARTICIONES),
    )
    direcciones = fila[2].multiselect(
        "Direccion", sorted(operaciones["direccion"].unique()),
        default=sorted(operaciones["direccion"].unique()),
    )
    resultado = fila[3].selectbox(
        "Resultado", ["Todas", "Solo ganadoras", "Solo perdedoras"]
    )

    filtrada = operaciones[
        operaciones["grupo"].isin(grupos)
        & operaciones["particion"].isin(particiones)
        & operaciones["direccion"].isin(direcciones)
    ]
    if resultado == "Solo ganadoras":
        filtrada = filtrada[filtrada["pnl_neto"] > 0]
    elif resultado == "Solo perdedoras":
        filtrada = filtrada[filtrada["pnl_neto"] <= 0]

    resumen = st.columns(4)
    resumen[0].metric("Operaciones", f"{len(filtrada)}")
    resumen[1].metric(
        "P&L neto", components.formatear_metrica(filtrada["pnl_neto"].sum(), "dolares")
    )
    resumen[2].metric(
        "Tasa de acierto",
        components.formatear_metrica(
            (filtrada["pnl_neto"] > 0).mean() if len(filtrada) else np.nan,
            "porcentaje",
        ),
    )
    resumen[3].metric(
        "Dias habiles promedio",
        f"{filtrada['dias_habiles'].mean():.0f}" if len(filtrada) else "n/d",
    )

    st.dataframe(
        filtrada.assign(
            fecha_entrada=filtrada["fecha_entrada"].dt.date,
            fecha_salida=filtrada["fecha_salida"].dt.date,
        )[
            ["grupo", "direccion", "fecha_entrada", "fecha_salida", "particion",
             "dias_habiles", "z_entrada", "z_salida", "pnl_neto", "razon_salida"]
        ].style.format(
            {"z_entrada": "{:.2f}", "z_salida": "{:.2f}", "pnl_neto": "${:,.0f}"}
        ),
        use_container_width=True,
        height=330,
    )
    return filtrada


def _distribucion_de_retornos(operaciones: pd.DataFrame) -> None:
    """Elemento 5: histograma de P&L por operacion con normal superpuesta."""
    st.subheader("Distribucion de retornos por operacion")
    if len(operaciones) < 3:
        st.info("Se necesitan al menos tres operaciones para el histograma.")
        return

    valores = operaciones["pnl_neto"].to_numpy(dtype=float)
    media, desviacion = float(np.mean(valores)), float(np.std(valores, ddof=1))

    figura = go.Figure()
    figura.add_trace(
        go.Histogram(
            x=valores, nbinsx=28, histnorm="probability density",
            name="P&L por operacion", marker_color=components.PALETA[0],
            opacity=0.7,
            hovertemplate="P&L: $%{x:,.0f}<br>densidad: %{y:.2e}<extra></extra>",
        )
    )
    rejilla = np.linspace(valores.min(), valores.max(), 300)
    densidad = np.exp(-0.5 * ((rejilla - media) / desviacion) ** 2) / (
        desviacion * np.sqrt(2.0 * np.pi)
    )
    figura.add_trace(
        go.Scatter(
            x=rejilla, y=densidad, name="Normal ajustada",
            line=dict(color="#b2182b", width=2.4),
            hovertemplate="$%{x:,.0f}<br>densidad: %{y:.2e}<extra></extra>",
        )
    )
    figura.add_vline(x=0.0, line=dict(color="black", width=1, dash="dot"))
    figura.update_layout(
        title=(
            f"{len(valores)} operaciones | media ${media:,.0f} | "
            f"desviacion ${desviacion:,.0f}"
        ),
        xaxis_title="P&L neto por operacion (USD)", yaxis_title="Densidad",
        height=430, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
    )
    st.plotly_chart(figura, use_container_width=True)

    sesgo = float(pd.Series(valores).skew())
    curtosis = float(pd.Series(valores).kurtosis())
    st.caption(
        f"Sesgo {sesgo:+.2f}, exceso de curtosis {curtosis:+.2f}. La normal es "
        "una referencia visual, no un supuesto del modelo: el backtest no asume "
        "normalidad en ninguna parte."
    )


def render() -> None:
    """Dibuja la vista ejecutiva completa."""
    st.header("Vista ejecutiva")
    st.caption(
        "Para un portfolio manager o un comite de inversion: resultado, costos "
        "y riesgo del portafolio consolidado."
    )
    _curva_de_capital()
    st.divider()
    _tarjeta_de_desempeno()
    st.divider()
    _atribucion_de_costos()
    st.divider()
    operaciones = _bitacora_de_operaciones()
    st.divider()
    _distribucion_de_retornos(operaciones)
