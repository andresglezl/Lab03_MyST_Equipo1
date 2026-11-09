"""Ingesta de precios diarios, con un snapshot congelado y versionado.

Por que existe el snapshot
--------------------------
yfinance NO devuelve la misma serie historica en dos fechas distintas. El cierre
ajustado se recalcula hacia atras cada vez que una empresa paga un dividendo o
hace un split: el factor de ajuste de TODA la historia previa cambia. Una
descarga hecha un mes despues de la que produjo este reporte daria precios
distintos para 2013, y con ellos vectores de Johansen distintos, umbrales
distintos y cifras distintas a las que el reporte afirma.

Para que el proyecto sea reproducible de verdad, la descarga se congela una sola
vez en `data/precios_snapshot.parquet`, ese archivo se versiona en el
repositorio, y `python main.py` lo usa por defecto. Quien clone el repositorio
obtiene exactamente las cifras del reporte, sin red y sin depender de que
yfinance siga respondiendo.

`python main.py --forzar-descarga` vuelve a bajar de yfinance y sobrescribe el
snapshot. Es la unica forma de actualizarlo, y hacerlo cambia los resultados: se
documenta en el README.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

import config


def ruta_snapshot() -> Path:
    """Ruta del snapshot congelado de precios crudos."""
    return config.RUTA_SNAPSHOT


def _descargar_de_yfinance(tickers: list[str], inicio: str, fin: str) -> pd.DataFrame:
    """Descarga precios de cierre ajustados desde yfinance.

    Se usa `auto_adjust=True` para obtener el cierre ajustado por splits y
    dividendos. El ajuste importa para la cointegracion: una serie sin ajustar
    tiene saltos de nivel en cada split que ninguna relacion de equilibrio
    puede absorber.

    Args:
        tickers: lista de simbolos.
        inicio: fecha inicial 'YYYY-MM-DD' (inclusiva).
        fin: fecha final 'YYYY-MM-DD' (inclusiva).

    Returns:
        DataFrame con indice de fechas y una columna por ticker.

    Raises:
        RuntimeError: si yfinance no devuelve datos.
    """
    # yfinance trata `end` como exclusivo; se agrega un dia para que `fin`
    # quede incluido en el panel.
    fin_exclusivo = (pd.Timestamp(fin) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    crudo = yf.download(
        tickers,
        start=inicio,
        end=fin_exclusivo,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if crudo is None or crudo.empty:
        raise RuntimeError("yfinance no devolvio datos para el universo solicitado.")

    precios = crudo["Close"] if isinstance(crudo.columns, pd.MultiIndex) else crudo[["Close"]]
    if not isinstance(crudo.columns, pd.MultiIndex):
        precios.columns = tickers
    precios.index = pd.to_datetime(precios.index).tz_localize(None).normalize()
    precios.index.name = "fecha"
    return precios.sort_index()[sorted(precios.columns)]


def _snapshot_cubre(
    snapshot: pd.DataFrame, tickers: list[str], inicio: str, fin: str
) -> bool:
    """Indica si el snapshot contiene los tickers y la ventana pedidos."""
    return (
        set(tickers).issubset(snapshot.columns)
        and snapshot.index.min() <= pd.Timestamp(inicio)
        and snapshot.index.max() >= pd.Timestamp(fin)
    )


def cargar_precios(
    tickers: list[str],
    inicio: str = config.FECHA_INICIO,
    fin: str = config.FECHA_FIN,
    forzar_descarga: bool = False,
) -> pd.DataFrame:
    """Devuelve el panel de precios, leyendo el snapshot congelado por defecto.

    Args:
        tickers: simbolos a cargar.
        inicio: fecha inicial 'YYYY-MM-DD'.
        fin: fecha final 'YYYY-MM-DD'.
        forzar_descarga: si True ignora el snapshot, descarga de yfinance y
            sobrescribe el snapshot. Cambia los resultados del reporte.

    Returns:
        DataFrame de precios con indice de fechas y columnas ordenadas.

    Raises:
        FileNotFoundError: si no hay snapshot y no se pidio descarga.
    """
    destino = ruta_snapshot()
    pedidos = sorted(tickers)

    if not forzar_descarga:
        if not destino.exists():
            raise FileNotFoundError(
                f"No existe el snapshot {destino.name}. Corra "
                "`python main.py --forzar-descarga` para crearlo desde yfinance."
            )
        snapshot = pd.read_parquet(destino)
        if not _snapshot_cubre(snapshot, pedidos, inicio, fin):
            raise FileNotFoundError(
                f"El snapshot {destino.name} no cubre los tickers o la ventana "
                "pedidos. Corra `python main.py --forzar-descarga` para "
                "regenerarlo."
            )
        return snapshot.loc[pd.Timestamp(inicio) : pd.Timestamp(fin), pedidos].copy()

    precios = _descargar_de_yfinance(pedidos, inicio, fin)
    destino.parent.mkdir(parents=True, exist_ok=True)
    precios.to_parquet(destino)
    return precios
