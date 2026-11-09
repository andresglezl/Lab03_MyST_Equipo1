"""Vista tecnica del dashboard (seccion 3.8).

Disenada para un investigador cuantitativo o un gestor de riesgos. Los siete
elementos que exige el PDF:

1. Evolucion del spread S_t con las senales de entrada y salida anotadas.
2. Evolucion del vector de pesos: cada componente de beta_t, una linea por activo.
3. Valores del primer eigenvector a lo largo del tiempo (Johansen movil).
4. Ganancia de Kalman en el tiempo para ambos filtros.
5. Serie del z-score con las lineas de umbral de entrada y salida.
6. Diagnosticos de convergencia: varianza posterior de la estimacion del estado.
7. Sharpe movil de 252 dias a lo largo de toda la muestra.

Se agrega la tabla de grupos probados y las contribuciones al riesgo, que son
material de nivel B y de la seccion 3.2 que un gestor de riesgos va a querer ver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from dashboard import components, data_access

#: Rango vertical de la grafica del eigenvector movil.
LIMITE_EIGENVECTOR = (-3.0, 3.0)


def _spread_y_senales(grupo: str, umbrales: dict) -> None:
    """Elemento 1: spread con entradas y salidas anotadas."""
    st.subheader("1. Spread del portafolio con senales")
    senales = data_access.senales(grupo)
    filtro2 = data_access.filtro2(grupo)

    figura = go.Figure()
    figura.add_trace(
        go.Scatter(
            x=senales.index, y=filtro2["spread"], name="Spread S",
            line=dict(color="#2c3e50", width=1.1),
            hovertemplate="%{x|%Y-%m-%d}<br>S=%{y:.3f}<extra></extra>",
        )
    )
    figura.add_trace(
        go.Scatter(
            x=senales.index, y=filtro2["mu_predicha"], name="Media latente",
            line=dict(color="#8e44ad", width=1.3, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>mu=%{y:.3f}<extra></extra>",
        )
    )
    for evento, nombre, simbolo, color in (
        ("entrada_larga", "Entrada larga", "triangle-up", "#1b7837"),
        ("entrada_corta", "Entrada corta", "triangle-down", "#b2182b"),
        ("salida", "Salida", "circle", "#2166ac"),
    ):
        marcas = senales[senales["evento"] == evento]
        figura.add_trace(
            go.Scatter(
                x=marcas.index, y=filtro2.loc[marcas.index, "spread"],
                name=nombre, mode="markers",
                marker=dict(symbol=simbolo, size=10, color=color),
                hovertemplate="%{x|%Y-%m-%d}<br>S=%{y:.3f}<extra></extra>",
            )
        )
    components.sombrear_particiones(figura, senales["particion"])
    components.aplicar_formato(
        figura, f"Spread y senales — {grupo}", "Spread (USD)", altura=460
    )
    st.plotly_chart(figura, use_container_width=True)


def _vector_de_pesos(grupo: str, tickers: list[str], numeraria: str) -> None:
    """Elemento 2: cada componente de beta_t, una linea por activo."""
    st.subheader("2. Evolucion del vector de pesos")
    serie = data_access.filtro1(grupo)
    senales = data_access.senales(grupo)

    figura = go.Figure()
    for posicion, ticker in enumerate(tickers):
        figura.add_trace(
            go.Scatter(
                x=serie.index, y=serie[f"beta_{ticker}"],
                name=f"{ticker}" + (" (numeraria)" if ticker == numeraria else ""),
                line=dict(color=components.PALETA[posicion], width=1.3),
                hovertemplate="%{x|%Y-%m-%d}<br>beta=%{y:.4f}<extra></extra>",
            )
        )
    figura.add_hline(y=0.0, line=dict(color="black", width=1, dash="dot"))
    components.sombrear_particiones(figura, senales["particion"])
    components.aplicar_formato(
        figura, f"Vector de pesos del filtro 1 (estado predicho) — {grupo}", "Peso"
    )
    st.plotly_chart(figura, use_container_width=True)
    st.caption(
        "La pata numeraria vale exactamente 1 en toda la muestra: no es variable "
        "de estado, se impone al reconstruir beta. Un peso que cruza cero "
        "significa que la pata cambio de lado en la relacion de equilibrio; los "
        "cruces estan listados en artifacts/kalman/cruces_por_cero_beta.csv."
    )


def _primer_eigenvector(grupo: str, tickers: list[str]) -> None:
    """Elemento 3: primer eigenvector de Johansen reestimado en ventanas."""
    st.subheader("3. Primer eigenvector de Johansen movil")
    tabla = data_access.johansen_movil(grupo)

    figura = go.Figure()
    # Las ventanas en que la relacion no cointegra al 95% se sombrean: ahi el
    # eigenvector no define un equilibrio y sus componentes no son interpretables.
    sin_cointegrar = ~tabla["cointegra_95"].to_numpy(dtype=bool)
    if sin_cointegrar.any():
        cambios = np.flatnonzero(np.diff(sin_cointegrar.astype(int)) != 0) + 1
        for bloque in np.split(np.arange(len(tabla)), cambios):
            if len(bloque) and sin_cointegrar[bloque[0]]:
                figura.add_vrect(
                    x0=tabla.index[bloque[0]], x1=tabla.index[bloque[-1]],
                    fillcolor="#b2182b", opacity=0.07, layer="below", line_width=0,
                )
    for posicion, ticker in enumerate(tickers):
        figura.add_trace(
            go.Scatter(
                x=tabla.index, y=tabla[f"eigvec_{ticker}"], name=ticker,
                line=dict(color=components.PALETA[posicion], width=1.4),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
            )
        )
    figura.add_trace(
        go.Scatter(
            x=tabla.index, y=tabla["traza_r0"], name="Traza (r=0)", yaxis="y2",
            line=dict(color="gray", width=1, dash="dot"),
            hovertemplate="%{x|%Y-%m-%d}<br>traza=%{y:.1f}<extra></extra>",
        )
    )
    figura.add_hline(
        y=float(tabla["traza_critico_95"].iloc[0]), yref="y2",
        line=dict(color="gray", width=1, dash="dash"),
        annotation_text="critico 95%", annotation_position="bottom right",
    )
    components.aplicar_formato(
        figura,
        f"Primer eigenvector re-anclado, ventana {config.VENTANA_JOHANSEN_MOVIL} d — {grupo}",
        "Componente",
    )
    figura.update_layout(
        yaxis=dict(range=list(LIMITE_EIGENVECTOR)),
        yaxis2=dict(title="Estadistico de traza", overlaying="y", side="right",
                    showgrid=False),
    )
    st.plotly_chart(figura, use_container_width=True)

    fraccion = float(tabla["cointegra_95"].mean())
    fuera_de_rango = int(
        (tabla[[f"eigvec_{t}" for t in tickers]].abs() > LIMITE_EIGENVECTOR[1])
        .any(axis=1)
        .sum()
    )
    st.caption(
        f"La ventana movil detecta cointegracion al 95% en el {fraccion:.1%} de "
        "las reestimaciones; las ventanas sombreadas en rojo son las que no. "
        f"El eje vertical se acota a {LIMITE_EIGENVECTOR}: el re-anclaje divide "
        "entre la componente de la pata numeraria y, cuando esa componente se "
        f"acerca a cero, el cociente explota ({fuera_de_rango} ventanas quedan "
        "fuera de rango). Comparar esta trayectoria con la del filtro 1 responde "
        "si el filtro se mueve porque la relacion cambio o porque Q lo deja "
        "perseguir ruido."
    )


def _ganancias_de_kalman(grupo: str) -> None:
    """Elemento 4: ganancia de Kalman de ambos filtros."""
    st.subheader("4. Ganancia de Kalman de ambos filtros")
    filtro1 = data_access.filtro1(grupo)
    filtro2 = data_access.filtro2(grupo)

    izquierda, derecha = st.columns(2)
    with izquierda:
        figura = go.Figure()
        columnas = [
            c for c in filtro1.columns
            if c.startswith("ganancia_") and c != "ganancia_norma"
        ]
        for posicion, columna in enumerate(columnas):
            figura.add_trace(
                go.Scatter(
                    x=filtro1.index, y=filtro1[columna],
                    name=columna.replace("ganancia_", ""),
                    line=dict(color=components.PALETA[posicion], width=1.1),
                    hovertemplate="%{x|%Y-%m-%d}<br>K=%{y:.3e}<extra></extra>",
                )
            )
        components.aplicar_formato(
            figura, "Filtro 1: ganancia por componente", "Ganancia K", 380
        )
        st.plotly_chart(figura, use_container_width=True)
    with derecha:
        figura = go.Figure()
        figura.add_trace(
            go.Scatter(
                x=filtro2.index, y=filtro2["ganancia_filtro2"],
                name="Ganancia del filtro 2", line=dict(color="#8e44ad", width=1.3),
                hovertemplate="%{x|%Y-%m-%d}<br>K=%{y:.5f}<extra></extra>",
            )
        )
        components.aplicar_formato(
            figura, "Filtro 2: ganancia de la media latente", "Ganancia K", 380
        )
        st.plotly_chart(figura, use_container_width=True)

    st.caption(
        "La ganancia es un cociente de incertidumbres: "
        "K_t = Sigma_pred H' / (H Sigma_pred H' + R). Cuanto mas chica, mas "
        "ignora el filtro la observacion del dia. La del filtro 2 converge a su "
        "valor estacionario K_inf porque su razon Q2/R2 se mantiene constante."
    )


def _zscore(grupo: str, umbrales: dict) -> None:
    """Elemento 5: z-score con las lineas de umbral."""
    st.subheader("5. z-score con umbrales")
    senales = data_access.senales(grupo)

    figura = go.Figure()
    figura.add_trace(
        go.Scatter(
            x=senales.index, y=senales["z"], name="z-score",
            line=dict(color="#2c3e50", width=1.0),
            hovertemplate="%{x|%Y-%m-%d}<br>z=%{y:.3f}<extra></extra>",
        )
    )
    for nombre, valor, color, trazo in (
        ("Entrada", umbrales["z_entrada"], "#b2182b", "dash"),
        ("Salida", umbrales["z_salida"], "#1b7837", "dot"),
        ("Stop", umbrales["z_stop"], "black", "dashdot"),
    ):
        for signo in (1, -1):
            # Pasar annotation_text=None no suprime la anotacion: Plotly la crea
            # con su texto por defecto ("new text"). Hay que no pasar el
            # argumento en absoluto cuando no se quiere etiqueta.
            if signo == 1:
                figura.add_hline(
                    y=valor, line=dict(color=color, width=1.1, dash=trazo),
                    annotation_text=f"{nombre} {valor:+.2f}",
                    annotation_position="right",
                )
            else:
                figura.add_hline(
                    y=-valor, line=dict(color=color, width=1.1, dash=trazo)
                )
    components.sombrear_particiones(figura, senales["particion"])
    components.aplicar_formato(
        figura,
        f"z-score ex-ante — {grupo} (entrada {umbrales['z_entrada']}, "
        f"salida {umbrales['z_salida']}, stop {umbrales['z_stop']}, "
        f"tenencia maxima {umbrales['tenencia_maxima']} d)",
        "z-score",
        460,
    )
    st.plotly_chart(figura, use_container_width=True)


def _convergencia(grupo: str) -> None:
    """Elemento 6: varianza posterior del estado de ambos filtros."""
    st.subheader("6. Diagnosticos de convergencia de los filtros")
    filtro1 = data_access.filtro1(grupo)
    filtro2 = data_access.filtro2(grupo)

    izquierda, derecha = st.columns(2)
    with izquierda:
        figura = go.Figure()
        for posicion, columna in enumerate(
            [c for c in filtro1.columns if c.startswith("var_post_")]
        ):
            figura.add_trace(
                go.Scatter(
                    x=filtro1.index, y=filtro1[columna],
                    name=columna.replace("var_post_", ""),
                    line=dict(color=components.PALETA[posicion], width=1.1),
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3e}<extra></extra>",
                )
            )
        # La linea del burn-in se dibuja como forma y la etiqueta se agrega por
        # separado: `add_vline` con `annotation_text` sobre un eje de fechas
        # intenta promediar las coordenadas de la anotacion y falla en esta
        # combinacion de plotly y pandas.
        fin_burn_in = filtro1.index[config.BURN_IN_DIAS]
        figura.add_shape(
            type="line", x0=fin_burn_in, x1=fin_burn_in, y0=0, y1=1,
            yref="paper", line=dict(color="black", width=1, dash="dot"),
        )
        figura.add_annotation(
            x=fin_burn_in, y=1.0, yref="paper", yanchor="bottom",
            text=f"fin del burn-in ({config.BURN_IN_DIAS} d)",
            showarrow=False, font=dict(size=10),
        )
        components.aplicar_formato(
            figura, "Filtro 1: varianza posterior del estado",
            "Varianza posterior", 380,
        )
        figura.update_yaxes(type="log")
        st.plotly_chart(figura, use_container_width=True)
    with derecha:
        figura = go.Figure()
        figura.add_trace(
            go.Scatter(
                x=filtro2.index, y=filtro2["var_posterior_mu"],
                name="Var posterior de mu", line=dict(color="#8e44ad", width=1.3),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.4f}<extra></extra>",
            )
        )
        figura.add_trace(
            go.Scatter(
                x=filtro2.index, y=filtro2["sigma_predicha"],
                name="Sigma predicha del spread", yaxis="y2",
                line=dict(color="#d9a300", width=1.3),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.3f}<extra></extra>",
            )
        )
        components.aplicar_formato(
            figura, "Filtro 2: incertidumbre de la media y varianza adaptativa",
            "Varianza posterior de mu", 380,
        )
        figura.update_layout(
            yaxis2=dict(title="Sigma del spread", overlaying="y", side="right",
                        showgrid=False)
        )
        st.plotly_chart(figura, use_container_width=True)

    st.caption(
        "La varianza posterior del filtro 1 cae rapido y se estabiliza: por eso "
        f"el burn-in de {config.BURN_IN_DIAS} dias es suficiente. La sigma del "
        "filtro 2 crece con el nivel de los precios, que es precisamente lo que "
        "la varianza adaptativa esta diseniada para seguir."
    )


def _sharpe_movil() -> None:
    """Elemento 7: Sharpe movil de 252 dias de toda la muestra."""
    st.subheader("7. Sharpe movil de 252 dias")
    tabla = data_access.sharpe_movil()
    series = [c for c in tabla.columns if c != "particion"]

    elegidas = st.multiselect(
        "Series", series, default=["consolidado_risk_parity"]
    )
    figura = go.Figure()
    for posicion, columna in enumerate(elegidas):
        figura.add_trace(
            go.Scatter(
                x=tabla.index, y=tabla[columna], name=columna,
                line=dict(color=components.PALETA[posicion % len(components.PALETA)],
                          width=1.3),
                hovertemplate="%{x|%Y-%m-%d}<br>Sharpe=%{y:.3f}<extra></extra>",
            )
        )
    figura.add_hline(y=0.0, line=dict(color="black", width=1, dash="dot"))
    components.sombrear_particiones(figura, tabla["particion"])
    components.aplicar_formato(
        figura, "Sharpe movil de 252 dias habiles", "Sharpe anualizado", 440
    )
    st.plotly_chart(figura, use_container_width=True)


def _nivel_b() -> None:
    """Material de nivel B: contribuciones al riesgo y correlacion de spreads."""
    st.subheader("Nivel B: Risk Parity entre grupos")
    pesos = data_access.pesos_risk_parity()
    correlacion = data_access.correlacion_spreads()

    izquierda, derecha = st.columns([0.55, 0.45])
    with izquierda:
        figura = go.Figure()
        figura.add_trace(
            go.Bar(
                x=pesos["grupo"], y=pesos["contribucion_riesgo_rp"] * 100,
                name="Risk Parity", marker_color=components.PALETA[0],
                hovertemplate="%{x}<br>%{y:.4f}%<extra></extra>",
            )
        )
        figura.add_trace(
            go.Bar(
                x=pesos["grupo"], y=pesos["contribucion_riesgo_iguales"] * 100,
                name="Pesos iguales", marker_color=components.PALETA[4],
                hovertemplate="%{x}<br>%{y:.4f}%<extra></extra>",
            )
        )
        figura.add_hline(
            y=100.0 / len(pesos), line=dict(color="black", width=1.3, dash="dash"),
            annotation_text="objetivo 1/G",
        )
        figura.update_layout(
            barmode="group", title="Contribucion al riesgo por grupo",
            yaxis_title="Contribucion (%)", height=400, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
        )
        st.plotly_chart(figura, use_container_width=True)
    with derecha:
        figura = go.Figure(
            go.Heatmap(
                z=correlacion.to_numpy(), x=correlacion.columns,
                y=correlacion.index, colorscale="RdBu_r", zmin=-1, zmax=1,
                text=correlacion.round(4).to_numpy(), texttemplate="%{text}",
                hovertemplate="%{y} / %{x}: %{z:.4f}<extra></extra>",
            )
        )
        figura.update_layout(
            title="Correlacion entre spreads", height=400, template="plotly_white"
        )
        st.plotly_chart(figura, use_container_width=True)

    st.dataframe(
        pesos[["grupo", "vol_anual_spread", "peso_risk_parity", "peso_iguales",
               "contribucion_riesgo_rp", "contribucion_riesgo_iguales"]].style.format(
            {
                "vol_anual_spread": "{:.2%}",
                "peso_risk_parity": "{:.4f}",
                "peso_iguales": "{:.4f}",
                "contribucion_riesgo_rp": "{:.8f}",
                "contribucion_riesgo_iguales": "{:.8f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("**Atribucion del drawdown maximo del consolidado**")
    st.dataframe(
        data_access.atribucion_drawdown()[
            ["grupo", "pnl_neto_en_ventana", "fraccion_de_la_caida", "fecha_pico",
             "fecha_valle", "drawdown"]
        ].style.format(
            {
                "pnl_neto_en_ventana": "${:,.0f}",
                "fraccion_de_la_caida": "{:.2%}",
                "drawdown": "{:.2%}",
            }
        ),
        use_container_width=True,
    )


def _diagnostico_y_seleccion() -> None:
    """Tablas de diagnostico de filtros, seleccion y rebalanceo."""
    with st.expander("Diagnostico de los filtros (varianza y autocorrelacion)"):
        st.dataframe(data_access.diagnostico_filtros(), use_container_width=True)
        st.warning(
            "La autocorrelacion de las innovaciones NO es baja (rho_1 entre 0.97 "
            "y 0.99) y no puede serlo: las innovaciones del filtro 1 son el "
            "spread, que es persistente por construccion con vida media de ~36 "
            "dias. La columna phi_implicado_por_vida_media muestra que ese valor "
            "es el que el modelo predice. La blancura que si se exige es la de "
            "los residuales del VECM, en la fila vecm_ect, y se cumple."
        )
    with st.expander("Grupos probados en el embudo de seleccion"):
        manifiesto = data_access.manifiesto()
        resumen = manifiesto["etapas"]["cointegracion"]
        columnas = st.columns(3)
        columnas[0].metric("Grupos evaluados (m)", resumen["m_grupos_evaluados"])
        columnas[1].metric(
            "Falsos positivos esperados", resumen["falsos_positivos_esperados"]
        )
        columnas[2].metric("Grupos que pasaron", resumen["n_grupos_que_pasaron"])
        st.dataframe(data_access.grupos_probados(), use_container_width=True)
    with st.expander("Rebalanceo diario contra banda de no-rebalanceo"):
        st.dataframe(data_access.comparacion_rebalanceo(), use_container_width=True)


def render() -> None:
    """Dibuja la vista tecnica completa."""
    st.header("Vista tecnica")
    st.caption(
        "Para un investigador cuantitativo o un gestor de riesgos: mecanica de "
        "los filtros, estabilidad de la relacion de equilibrio y diagnosticos."
    )

    grupos = data_access.grupos_operados()
    etiquetas = [g["grupo"] for g in grupos]
    elegido = st.selectbox("Grupo cointegrado", etiquetas)
    metadatos = next(g for g in grupos if g["grupo"] == elegido)

    cabecera = st.columns(4)
    cabecera[0].metric("Pata numeraria", metadatos["pata_numeraria"])
    cabecera[1].metric(
        "Vida media del spread", f"{metadatos['vecm']['vida_media_dias']:.1f} d"
    )
    cabecera[2].metric("beta'alpha", f"{metadatos['vecm']['beta_alpha']:+.5f}")
    cabecera[3].metric("Rezagos del VECM", metadatos["vecm"]["k_rezagos"])

    umbrales = metadatos["umbrales"]
    _spread_y_senales(elegido, umbrales)
    st.divider()
    _vector_de_pesos(elegido, metadatos["tickers"], metadatos["pata_numeraria"])
    st.divider()
    _primer_eigenvector(elegido, metadatos["tickers"])
    st.divider()
    _ganancias_de_kalman(elegido)
    st.divider()
    _zscore(elegido, umbrales)
    st.divider()
    _convergencia(elegido)
    st.divider()
    _sharpe_movil()
    st.divider()
    _nivel_b()
    st.divider()
    _diagnostico_y_seleccion()
