"""Las nueve figuras obligatorias del reporte (seccion 3.10).

Se generan con matplotlib y se guardan en `artifacts/figures/` como PNG para el
reporte. El dashboard NO usa estos PNG: reconstruye las mismas vistas de forma
interactiva leyendo los mismos artifacts, como pide la seccion 3.8.

Todas las figuras llevan titulo, ejes etiquetados y leyenda legible, porque el
criterio G de la rubrica lo evalua explicitamente.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: el pipeline corre sin display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

#: Colores de las particiones, consistentes en todas las figuras.
COLOR_PARTICION = {
    "entrenamiento": "#1b7837",
    "prueba": "#d9a300",
    "validacion": "#b2182b",
}
DPI = 140

#: Rango vertical de la figura del eigenvector movil. Ver fig04 para el motivo.
LIMITE_EIGENVECTOR = (-3.0, 3.0)


def _sombrear_particiones(eje, etiquetas: pd.Series) -> None:
    """Sombrea el fondo del eje segun la particion de cada fecha."""
    for particion, color in COLOR_PARTICION.items():
        fechas = etiquetas.index[etiquetas.values == particion]
        if len(fechas) == 0:
            continue
        eje.axvspan(fechas[0], fechas[-1], color=color, alpha=0.07, zorder=0)
        eje.text(
            fechas[len(fechas) // 2],
            eje.get_ylim()[1],
            particion,
            ha="center",
            va="top",
            fontsize=8,
            color=color,
        )


def _guardar(figura, nombre: str) -> str:
    """Guarda una figura en `artifacts/figures/` y devuelve su ruta relativa."""
    import artifacts_io

    destino = artifacts_io.ruta("figures", nombre)
    figura.tight_layout()
    figura.savefig(destino, dpi=DPI)
    plt.close(figura)
    return f"figures/{nombre}"


def fig01_precios(precios: pd.DataFrame, etiquetas: pd.Series, grupos: dict) -> str:
    """Figura 1: series de precios individuales de los activos de cada grupo."""
    figura, ejes = plt.subplots(len(grupos), 1, figsize=(11, 4 * len(grupos)))
    ejes = np.atleast_1d(ejes)
    for eje, (grupo, tickers) in zip(ejes, grupos.items()):
        for ticker in tickers:
            eje.plot(precios.index, precios[ticker], linewidth=1.0, label=ticker)
        eje.set_title(f"Precios de cierre ajustado — grupo {grupo}")
        eje.set_ylabel("Precio (USD)")
        eje.legend(ncol=len(tickers), fontsize=9)
        eje.grid(alpha=0.3)
        _sombrear_particiones(eje, etiquetas)
    ejes[-1].set_xlabel("Fecha")
    return _guardar(figura, "fig01_precios.png")


def fig02_spread_senales(senales: dict, etiquetas: pd.Series) -> str:
    """Figura 2: evolucion del spread con las senales de entrada y salida."""
    figura, ejes = plt.subplots(len(senales), 1, figsize=(11, 4 * len(senales)))
    ejes = np.atleast_1d(ejes)
    for eje, (grupo, serie) in zip(ejes, senales.items()):
        eje.plot(serie.index, serie["spread"], color="#2c3e50", linewidth=0.9, label="Spread $S_t$")
        eje.plot(serie.index, serie["mu_predicha"], color="#8e44ad", linewidth=1.0,
                 linestyle="--", label=r"Media latente $\hat\mu_{t|t-1}$")
        largas = serie[serie["evento"] == "entrada_larga"]
        cortas = serie[serie["evento"] == "entrada_corta"]
        salidas = serie[serie["evento"] == "salida"]
        eje.scatter(largas.index, largas["spread"], marker="^", s=45, color="#1b7837",
                    zorder=3, label="Entrada larga")
        eje.scatter(cortas.index, cortas["spread"], marker="v", s=45, color="#b2182b",
                    zorder=3, label="Entrada corta")
        eje.scatter(salidas.index, salidas["spread"], marker="o", s=22, color="#2166ac",
                    zorder=3, label="Salida")
        eje.set_title(f"Spread del portafolio y senales — grupo {grupo}")
        eje.set_ylabel("Spread (USD)")
        eje.legend(fontsize=8, ncol=5)
        eje.grid(alpha=0.3)
        _sombrear_particiones(eje, etiquetas)
    ejes[-1].set_xlabel("Fecha")
    return _guardar(figura, "fig02_spread_senales.png")


def fig03_vector_pesos(filtros1: dict, etiquetas: pd.Series) -> str:
    """Figura 3: evolucion del vector de pesos, una linea por activo."""
    figura, ejes = plt.subplots(len(filtros1), 1, figsize=(11, 4 * len(filtros1)))
    ejes = np.atleast_1d(ejes)
    for eje, (grupo, serie) in zip(ejes, filtros1.items()):
        columnas = [c for c in serie.columns if c.startswith("beta_")]
        for columna in columnas:
            eje.plot(serie.index, serie[columna], linewidth=1.0,
                     label=columna.replace("beta_", r"$\hat\beta$ "))
        eje.axhline(1.0, color="black", linewidth=0.7, linestyle=":",
                    label="Pata numeraria = 1")
        eje.set_title(r"Vector de pesos $\hat\beta_{t|t-1}$ del filtro 1 — grupo " + grupo)
        eje.set_ylabel("Peso")
        eje.legend(fontsize=8, ncol=len(columnas) + 1)
        eje.grid(alpha=0.3)
        _sombrear_particiones(eje, etiquetas)
    ejes[-1].set_xlabel("Fecha")
    return _guardar(figura, "fig03_vector_pesos.png")


def fig04_primer_eigenvector(moviles: dict) -> str:
    """Figura 4: componentes del primer eigenvector de Johansen movil."""
    figura, ejes = plt.subplots(len(moviles), 1, figsize=(11, 4 * len(moviles)))
    ejes = np.atleast_1d(ejes)
    for eje, (grupo, tabla) in zip(ejes, moviles.items()):
        columnas = [c for c in tabla.columns if c.startswith("eigvec_")]
        # Las ventanas donde la relacion no cointegra al 95% se sombrean: ahi el
        # eigenvector no define una relacion de equilibrio y sus componentes no
        # son interpretables.
        sin_cointegrar = ~tabla["cointegra_95"].to_numpy(dtype=bool)
        if sin_cointegrar.any():
            eje.fill_between(
                tabla.index, 0, 1, where=sin_cointegrar, transform=eje.get_xaxis_transform(),
                color="#b2182b", alpha=0.08, zorder=0,
                label="Ventana sin cointegracion al 95%",
            )
        for columna in columnas:
            eje.plot(tabla.index, tabla[columna], linewidth=1.1,
                     label=columna.replace("eigvec_", ""))
        eje.set_title(
            f"Primer eigenvector de Johansen movil (ventana "
            f"{config.VENTANA_JOHANSEN_MOVIL} d) — grupo {grupo}"
        )
        eje.set_ylabel("Componente (re-anclada)")
        # El re-anclaje divide entre la componente de la pata numeraria. En las
        # ventanas donde esa componente se acerca a cero el cociente explota
        # (picos de +-1000) y aplasta visualmente el resto de la serie, asi que
        # el eje se acota al rango donde los pesos son interpretables.
        eje.set_ylim(*LIMITE_EIGENVECTOR)
        eje.legend(fontsize=8, ncol=len(columnas))
        eje.grid(alpha=0.3)
        eje_gemelo = eje.twinx()
        eje_gemelo.plot(tabla.index, tabla["traza_r0"], color="gray", linewidth=0.8,
                        alpha=0.6, label="Traza r=0")
        eje_gemelo.axhline(tabla["traza_critico_95"].iloc[0], color="gray",
                           linestyle="--", linewidth=0.8)
        eje_gemelo.set_ylabel("Estadistico de traza", color="gray", fontsize=8)
        eje_gemelo.legend(fontsize=7, loc="lower right")
    ejes[-1].set_xlabel("Fecha de cierre de la ventana")
    return _guardar(figura, "fig04_primer_eigenvector.png")


def fig05_ganancias_kalman(filtros1: dict, filtros2: dict, etiquetas: pd.Series) -> str:
    """Figura 5: ganancia de Kalman de ambos filtros."""
    figura, ejes = plt.subplots(len(filtros1), 2, figsize=(13, 3.6 * len(filtros1)))
    ejes = np.atleast_2d(ejes)
    for fila, grupo in enumerate(filtros1):
        serie1, serie2 = filtros1[grupo], filtros2[grupo]
        izquierda = ejes[fila, 0]
        for columna in [c for c in serie1.columns if c.startswith("ganancia_")]:
            if columna == "ganancia_norma":
                continue
            izquierda.plot(serie1.index, serie1[columna], linewidth=0.9,
                           label=columna.replace("ganancia_", ""))
        izquierda.set_title(f"Filtro 1: ganancia $K_t$ por componente — {grupo}")
        izquierda.set_ylabel("Ganancia")
        izquierda.legend(fontsize=7)
        izquierda.grid(alpha=0.3)

        derecha = ejes[fila, 1]
        derecha.plot(serie2.index, serie2["ganancia_filtro2"], color="#8e44ad",
                     linewidth=1.0, label=r"$K_t^{(2)}$")
        derecha.set_title(f"Filtro 2: ganancia $K_t$ de la media latente — {grupo}")
        derecha.set_ylabel("Ganancia")
        derecha.legend(fontsize=8)
        derecha.grid(alpha=0.3)
    for eje in ejes[-1]:
        eje.set_xlabel("Fecha")
    return _guardar(figura, "fig05_ganancias_kalman.png")


def fig06_zscore(senales: dict, umbrales: dict, etiquetas: pd.Series) -> str:
    """Figura 6: serie del z-score con las lineas de umbral."""
    figura, ejes = plt.subplots(len(senales), 1, figsize=(11, 4 * len(senales)))
    ejes = np.atleast_1d(ejes)
    for eje, (grupo, serie) in zip(ejes, senales.items()):
        parametros = umbrales[grupo]
        eje.plot(serie.index, serie["z"], color="#2c3e50", linewidth=0.8, label="$z_t$")
        for signo in (1, -1):
            eje.axhline(signo * parametros["z_entrada"], color="#b2182b",
                        linestyle="--", linewidth=0.9,
                        label="Entrada" if signo == 1 else None)
            eje.axhline(signo * parametros["z_salida"], color="#1b7837",
                        linestyle=":", linewidth=0.9,
                        label="Salida" if signo == 1 else None)
            eje.axhline(signo * parametros["z_stop"], color="black",
                        linestyle="-.", linewidth=0.9,
                        label="Stop" if signo == 1 else None)
        eje.set_title(f"z-score ex-ante y umbrales — grupo {grupo}")
        eje.set_ylabel("$z_t$")
        eje.legend(fontsize=8, ncol=4)
        eje.grid(alpha=0.3)
        _sombrear_particiones(eje, etiquetas)
    ejes[-1].set_xlabel("Fecha")
    return _guardar(figura, "fig06_zscore.png")


def fig07_curvas_capital(
    consolidado: pd.DataFrame, pesos_iguales: pd.DataFrame, individuales: dict
) -> str:
    """Figura 7: curvas de capital de los tres tramos con drawdown superpuesto."""
    figura, (arriba, abajo) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    arriba.plot(consolidado.index, consolidado["capital"], color="#2166ac",
                linewidth=1.6, label="Consolidado Risk Parity")
    arriba.plot(pesos_iguales.index, pesos_iguales["capital"], color="#8e44ad",
                linewidth=1.1, linestyle="--", label="Consolidado pesos iguales")
    for grupo, serie in individuales.items():
        arriba.plot(serie.index, serie["capital"], linewidth=0.9, alpha=0.75,
                    label=f"{grupo} solo")
    arriba.axhline(config.CAPITAL_INICIAL, color="black", linewidth=0.7, linestyle=":",
                   label="Capital inicial")
    arriba.set_title("Curva de capital por particion (entrenamiento, prueba, validacion)")
    arriba.set_ylabel("Capital (USD)")
    arriba.legend(fontsize=8)
    arriba.grid(alpha=0.3)
    _sombrear_particiones(arriba, consolidado["particion"])

    abajo.fill_between(consolidado.index, consolidado["drawdown"] * 100, 0,
                       color="#b2182b", alpha=0.45)
    abajo.set_title("Drawdown del portafolio consolidado")
    abajo.set_ylabel("Drawdown (%)")
    abajo.set_xlabel("Fecha")
    abajo.grid(alpha=0.3)
    return _guardar(figura, "fig07_curvas_capital.png")


def fig08_distribucion_retornos(operaciones: pd.DataFrame) -> str:
    """Figura 8: histograma de P&L por operacion con una normal superpuesta."""
    figura, eje = plt.subplots(figsize=(10, 5.5))
    valores = operaciones["pnl_neto"].to_numpy(dtype=float)
    eje.hist(valores, bins=28, density=True, color="#2166ac", alpha=0.65,
             edgecolor="white", label="P&L por operacion")

    media, desviacion = float(np.mean(valores)), float(np.std(valores, ddof=1))
    rejilla = np.linspace(valores.min(), valores.max(), 300)
    densidad = (
        np.exp(-0.5 * ((rejilla - media) / desviacion) ** 2)
        / (desviacion * np.sqrt(2.0 * np.pi))
    )
    eje.plot(rejilla, densidad, color="#b2182b", linewidth=1.8,
             label=f"Normal ajustada ($\\mu$={media:,.0f}, $\\sigma$={desviacion:,.0f})")
    eje.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    eje.set_title(
        f"Distribucion del P&L neto por operacion ({len(valores)} operaciones, "
        "ambos grupos)"
    )
    eje.set_xlabel("P&L neto por operacion (USD)")
    eje.set_ylabel("Densidad")
    eje.legend(fontsize=9)
    eje.grid(alpha=0.3)
    return _guardar(figura, "fig08_distribucion_retornos.png")


def fig09_risk_parity(tabla_pesos: pd.DataFrame, correlacion: pd.DataFrame) -> str:
    """Figura 9 (nivel B): contribuciones al riesgo y correlacion entre spreads."""
    figura, (izquierda, derecha) = plt.subplots(1, 2, figsize=(12.5, 5))

    posiciones = np.arange(len(tabla_pesos))
    ancho = 0.35
    izquierda.bar(posiciones - ancho / 2, tabla_pesos["contribucion_riesgo_rp"] * 100,
                  ancho, color="#2166ac", label="Risk Parity")
    izquierda.bar(posiciones + ancho / 2,
                  tabla_pesos["contribucion_riesgo_iguales"] * 100, ancho,
                  color="#d9a300", label="Pesos iguales")
    izquierda.axhline(100.0 / len(tabla_pesos), color="black", linestyle="--",
                      linewidth=1.0, label="Objetivo $1/G$")
    izquierda.set_xticks(posiciones)
    izquierda.set_xticklabels(tabla_pesos["grupo"], fontsize=8)
    izquierda.set_title("Contribucion al riesgo por grupo")
    izquierda.set_ylabel("Contribucion al riesgo (%)")
    izquierda.legend(fontsize=8)
    izquierda.grid(alpha=0.3, axis="y")

    imagen = derecha.imshow(correlacion.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    derecha.set_xticks(range(len(correlacion)))
    derecha.set_yticks(range(len(correlacion)))
    derecha.set_xticklabels(correlacion.columns, fontsize=8, rotation=20, ha="right")
    derecha.set_yticklabels(correlacion.index, fontsize=8)
    for i in range(len(correlacion)):
        for j in range(len(correlacion)):
            derecha.text(j, i, f"{correlacion.iloc[i, j]:.3f}", ha="center",
                         va="center", fontsize=10)
    derecha.set_title("Correlacion entre los spreads de los grupos")
    figura.colorbar(imagen, ax=derecha, fraction=0.046)
    return _guardar(figura, "fig09_risk_parity.png")
