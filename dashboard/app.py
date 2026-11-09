"""Dashboard del Lab 03 MyST (seccion 3.8).

Se ejecuta localmente con un solo comando:

    streamlit run dashboard/app.py

Sirve a dos audiencias distintas en dos vistas separadas: la ejecutiva, para un
portfolio manager o un comite de inversion, y la tecnica, para un investigador
cuantitativo o un gestor de riesgos.

Lee EXCLUSIVAMENTE los archivos de `artifacts/` a traves de
`dashboard/data_access.py`. No importa ningun modulo de `backtest/`, `kalman/`,
`strategy/` ni `cointegration/`, de modo que es estructuralmente incapaz de
volver a correr el backtest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from dashboard import data_access, views_executive, views_technical  # noqa: E402

st.set_page_config(
    page_title="Lab 03 MyST — Arbitraje estadistico",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _barra_lateral() -> str:
    """Dibuja la barra lateral y devuelve la vista elegida."""
    st.sidebar.title("Lab 03 MyST — Equipo 1")
    st.sidebar.caption(
        "Arbitraje estadistico con cointegracion y filtros de Kalman · Nivel B"
    )
    vista = st.sidebar.radio(
        "Vista", ["Ejecutiva", "Tecnica"],
        captions=[
            "Portfolio manager / comite de inversion",
            "Investigador cuantitativo / gestor de riesgos",
        ],
    )

    manifiesto = data_access.manifiesto()
    corrida = manifiesto["corrida"]
    parametros = manifiesto["parametros"]
    datos = manifiesto["etapas"]["datos"]["panel"]

    st.sidebar.divider()
    st.sidebar.subheader("Corrida")
    st.sidebar.text(f"ID         {corrida['id']}")
    st.sidebar.text(f"Ejecutada  {corrida['fecha_ejecucion'][:10]}")
    st.sidebar.text(f"Semilla    {parametros['semilla']}")
    st.sidebar.text(f"Python     {corrida['python']}")

    st.sidebar.subheader("Muestra")
    st.sidebar.text(f"{datos['fecha_inicio']} a {datos['fecha_fin']}")
    st.sidebar.text(f"{datos['n_observaciones']:,} dias · {datos['n_tickers']} tickers")
    st.sidebar.text(f"{datos['anos_de_historia']} anos")

    st.sidebar.subheader("Parametros")
    st.sidebar.text(f"Capital     ${parametros['capital_inicial']:,.0f}")
    st.sidebar.text(f"Comision    {parametros['comision_por_pata']:.3%} / pata")
    st.sidebar.text(
        f"Prestamo    {parametros['tasa_prestamo_anual']:.2%} act/"
        f"{parametros['dias_devengo_prestamo']}"
    )
    st.sidebar.text(f"Exposicion  {parametros['exposicion_maxima']:.0%} consolidado")
    st.sidebar.text(f"Anualizac.  {parametros['dias_anualizacion']} dias")

    st.sidebar.subheader("Grupos operados")
    for grupo in data_access.grupos_operados():
        st.sidebar.text(f"{grupo['grupo']}")
        st.sidebar.caption(
            f"numeraria {grupo['pata_numeraria']} · vida media "
            f"{grupo['vecm']['vida_media_dias']:.0f} d"
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "Este dashboard solo LEE artifacts/. Para regenerarlos corra "
        "`python main.py`."
    )
    return vista


def main() -> None:
    """Punto de entrada del dashboard."""
    if not data_access.hay_artifacts():
        st.title("Lab 03 MyST — Equipo 1")
        st.error(
            "No se encontraron artifacts. El dashboard solo lee archivos ya "
            "generados y no ejecuta el backtest."
        )
        st.code("python main.py", language="bash")
        st.stop()

    vista = _barra_lateral()
    st.title("Arbitraje estadistico con cointegracion y filtros de Kalman")

    if vista == "Ejecutiva":
        views_executive.render()
    else:
        views_technical.render()


main()
