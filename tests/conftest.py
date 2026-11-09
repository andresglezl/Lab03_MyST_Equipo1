"""Fixtures compartidos: leen los artifacts que produjo `python main.py`.

Las pruebas NO vuelven a correr el backtest completo: leen `artifacts/`, igual
que el dashboard. La unica excepcion es la prueba de causalidad, que por su
naturaleza tiene que re-ejecutar la cadena de filtros sobre datos truncados, y
para eso usa el helper `recalcular_z` definido aqui.

Si los artifacts no existen, las pruebas se saltan con un mensaje que dice que
hay que correr el pipeline primero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import artifacts_io  # noqa: E402
import config  # noqa: E402
from data import splits  # noqa: E402
from kalman import filter1_weights, filter2_signal  # noqa: E402


def _exigir_artifacts() -> None:
    """Salta la sesion de pruebas si el pipeline no se ha corrido."""
    if not artifacts_io.ruta("manifest.json").exists():
        pytest.skip(
            "No hay artifacts. Corra `python main.py` antes de `pytest`.",
            allow_module_level=True,
        )


@pytest.fixture(scope="session")
def manifiesto() -> dict:
    """Manifiesto de la corrida."""
    _exigir_artifacts()
    return artifacts_io.leer_json("manifest.json")


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    """Panel de precios limpio, sin la columna de particion."""
    _exigir_artifacts()
    return artifacts_io.leer_serie("prices.parquet").drop(columns="particion")


@pytest.fixture(scope="session")
def etiquetas(panel: pd.DataFrame) -> pd.Series:
    """Serie de particiones por fecha."""
    return splits.etiquetar_particiones(panel.index)


@pytest.fixture(scope="session")
def fin_entrenamiento(panel: pd.DataFrame) -> pd.Timestamp:
    """Ultima fecha de la particion de entrenamiento del panel completo."""
    return splits.dividir_cronologico(panel.index)["entrenamiento"][-1]


@pytest.fixture(scope="session")
def grupos(manifiesto: dict) -> list[dict]:
    """Metadatos de los grupos operados, tomados del manifiesto."""
    return manifiesto["etapas"]["estrategia"]["grupos"]


@pytest.fixture(scope="session")
def grupos_cointegracion(manifiesto: dict) -> dict:
    """Vector beta y pata numeraria de cada grupo operado."""
    return {
        entrada["grupo"]: entrada
        for entrada in manifiesto["etapas"]["cointegracion"]["grupos_operados"]
    }


def recalcular_z(
    panel_hasta_t: pd.DataFrame,
    tickers: list[str],
    pata_numeraria: str,
    vida_media: float,
    fin_entrenamiento: pd.Timestamp,
    beta_johansen: np.ndarray,
) -> pd.Series:
    """Recalcula el z-score corriendo los dos filtros sobre un panel truncado.

    Reproduce exactamente la cadena causal del pipeline: calibra el filtro 1 con
    la particion de entrenamiento del panel recibido, corre la recursion, y
    alimenta el filtro 2 con el spread resultante.

    La fecha de corte de entrenamiento se recibe como argumento y es la del
    panel COMPLETO, no la que resultaria de aplicar el 60% al panel truncado.
    Esto es esencial: los cortes 60/20/20 son parametros congelados del
    experimento, y recalcularlos sobre la muestra truncada cambiaria el conjunto
    con el que se calibran Q, R y el estado inicial, con lo que la prueba
    compararia dos experimentos distintos en lugar de medir causalidad.

    Args:
        panel_hasta_t: panel de precios recortado hasta la fecha t inclusive.
        tickers: activos del grupo en el orden del vector beta.
        pata_numeraria: ticker numerario.
        vida_media: vida media del spread estimada en entrenamiento.
        fin_entrenamiento: ultima fecha de la particion de entrenamiento del
            panel completo.
        beta_johansen: vector cointegrante que fija el estado inicial del
            filtro 1, igual que en el pipeline.

    Returns:
        Serie del z-score sobre el panel truncado.
    """
    precios = panel_hasta_t[tickers]
    mascara = np.asarray(precios.index <= fin_entrenamiento)
    if mascara.sum() == 0:
        raise ValueError("El panel truncado no contiene datos de entrenamiento.")

    resultado_f1 = filter1_weights.ejecutar_filtro1(
        precios, precios.loc[mascara], tickers, pata_numeraria, beta_johansen
    )
    spread = resultado_f1["serie"]["spread"]
    operable = resultado_f1["serie"]["operable"]
    semilla = float(spread.loc[mascara & operable.to_numpy()].var(ddof=1))
    resultado_f2 = filter2_signal.ejecutar_filtro2(spread, vida_media, semilla)
    return resultado_f2["serie"]["z"]


def autocorrelacion(serie: np.ndarray, rezago: int = 1) -> float:
    """Autocorrelacion muestral de una serie a un rezago dado."""
    return float(np.corrcoef(serie[:-rezago], serie[rezago:])[0, 1])
