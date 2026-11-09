"""Parametros globales del Lab 03 MyST - Equipo 1.

Todas las constantes del laboratorio viven aqui para que cualquier decision de
diseno sea rastreable a un solo lugar y el pipeline sea reproducible.
Las referencias (seccion X.Y) apuntan a docs/Lab03.pdf.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibilidad (seccion 2.2: "La semilla aleatoria debe estar fijada")
# ---------------------------------------------------------------------------
SEMILLA = 42


def fijar_semilla(semilla: int = SEMILLA) -> None:
    """Fija la semilla de los generadores aleatorios del proceso.

    Se fija `PYTHONHASHSEED` ademas de `random` y `numpy` porque el orden de
    iteracion de algunos conjuntos influye en el orden en que se evaluan los
    grupos candidatos, y queremos que la tabla de grupos probados sea identica
    entre corridas.
    """
    os.environ["PYTHONHASHSEED"] = str(semilla)
    random.seed(semilla)
    np.random.seed(semilla)


# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent
DIR_ARTIFACTS = RAIZ / "artifacts"
DIR_DOCS = RAIZ / "docs"

# Snapshot congelado y VERSIONADO de los precios crudos. yfinance recalcula el
# cierre ajustado hacia atras con cada dividendo, asi que una descarga en otra
# fecha daria cifras distintas a las del reporte. Ver data/loader.py.
RUTA_SNAPSHOT = RAIZ / "data" / "precios_snapshot.parquet"

SUBDIRS_ARTIFACTS = (
    "selection",
    "kalman",
    "rolling",
    "strategy",
    "backtest",
    "riskparity",
    "performance",
    "figures",
)


def preparar_directorios() -> None:
    """Crea el arbol completo de `artifacts/` si no existe."""
    DIR_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS_ARTIFACTS:
        (DIR_ARTIFACTS / sub).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Ventana historica (seccion 3.1: 10-15 anos de datos diarios)
# ---------------------------------------------------------------------------
# La fecha de inicio la impone PBF, que cotiza desde 2012-12-13 y es el nombre
# mas joven del universo pre-registrado. Se usa UNA ventana comun para los 11
# tickers (no una por grupo) por dos razones:
#   1. Todos los grupos candidatos se prueban sobre la misma muestra, de modo
#      que sus estadisticos de Johansen son comparables y la seleccion no
#      premia al grupo que casualmente tenia mas historia.
#   2. El portafolio consolidado de Risk Parity (seccion 3.7) necesita una
#      sola rejilla de fechas alineada entre grupos.
# Ventana resultante: 2012-12-13 a 2026-11-06 = 13.90 anos.
FECHA_INICIO = "2012-12-13"
FECHA_FIN = "2026-11-06"  # ultima sesion completa al momento de fijar la corrida

# ---------------------------------------------------------------------------
# Division de datos (seccion 3.1: 60/20/20 cronologico, sin look-ahead)
# ---------------------------------------------------------------------------
FRACCION_ENTRENAMIENTO = 0.60
FRACCION_PRUEBA = 0.20
FRACCION_VALIDACION = 0.20
NOMBRES_PARTICIONES = ("entrenamiento", "prueba", "validacion")

# ---------------------------------------------------------------------------
# Parametros de costos y capital (seccion 3.1)
# ---------------------------------------------------------------------------
CAPITAL_INICIAL = 1_000_000.0
COMISION_POR_PATA = 0.00125  # 0.125% por operacion, por pata, entrada y salida
TASA_PRESTAMO_ANUAL = 0.0025  # 0.25% anual sobre el nocional en corto

# Convencion de devengo del prestamo: actual/365 sobre dias CALENDARIO entre
# barras consecutivas. Se cobra fin de semana y feriados (un corto los paga) y
# el costo de un ano completo con nocional constante es exactamente 0.25%.
DIAS_DEVENGO_PRESTAMO = 365

# Convencion de anualizacion de metricas: 252 dias habiles. Es deliberadamente
# DISTINTA de DIAS_DEVENGO_PRESTAMO: una mide carry de calendario, la otra
# escala momentos de retornos observados solo en dias de operacion.
DIAS_ANUALIZACION = 252

# Limite de exposicion: 80% del capital del portafolio CONSOLIDADO, medido como
# nocional BRUTO (suma de valores absolutos de todas las patas).
EXPOSICION_MAXIMA = 0.80

# ---------------------------------------------------------------------------
# Umbrales del proceso de seleccion (seccion 3.2)
# ---------------------------------------------------------------------------
CORRELACION_MINIMA = 0.70  # filtro inicial, heuristico
VENTANA_CORRELACION_MOVIL = 252  # 1 ano habil
PVALUE_MAXIMO = 0.05  # Engle-Granger y ADF sobre el spread

# Especificacion de Johansen. det_order=0 en statsmodels es una constante NO
# RESTRINGIDA: entra al VAR pero no se restringe a la relacion de
# cointegracion, de modo que el spread beta'P no se fuerza a tener media cero.
# Su nivel lo absorben el intercepto c_t del filtro 1 y la media latente mu_t
# del filtro 2. k_ar_diff=1 por parsimonia: se verifico que el resultado del
# test no cambia con k en {1, 2, 5} ni con k elegido por AIC, asi que el
# hallazgo no es un artefacto de la seleccion de rezagos.
JOHANSEN_DET_ORDER = 0
JOHANSEN_K_AR_DIFF = 1
JOHANSEN_NIVEL_CRITICO = 1  # columna de valores criticos: 0=90%, 1=95%, 2=99%

# Rango de vida media admisible para que un spread sea operable. Por debajo de
# 5 dias los costos de 0.125% por pata se comen la reversion; por encima de 126
# dias (6 meses) una posicion abierta consume capital demasiado tiempo y el
# riesgo de que la relacion se rompa domina.
VIDA_MEDIA_MINIMA = 5.0
VIDA_MEDIA_MAXIMA = 126.0

# Johansen movil para la figura 4 (primer eigenvector en el tiempo).
VENTANA_JOHANSEN_MOVIL = 504  # 2 anos habiles
PASO_JOHANSEN_MOVIL = 5  # una reestimacion por semana habil

# Nivel B: numero de grupos disjuntos que se operan en el portafolio consolidado.
N_GRUPOS_OBJETIVO = 2

# VECM. El orden de rezagos se elige como el MENOR k de la malla cuyos
# residuales de la forma reducida del termino de correccion de error pasan
# Ljung-Box al 5%. La prueba se evalua en k + LJUNGBOX_REZAGOS_EXTRA rezagos,
# es decir MAS ALLA de los rezagos ajustados: evaluarla dentro de los rezagos
# ajustados es circular, porque la regresion absorbe por construccion
# justamente esas autocorrelaciones y el p-value sale artificialmente cerca de 1.
MALLA_REZAGOS_VECM = (1, 2, 3, 5, 8, 10)
LJUNGBOX_REZAGOS_EXTRA = 10
LJUNGBOX_ALPHA = 0.05

# Malla del grid search de umbrales del z-score (solo entrenamiento).
# z_salida NO incluye 0.0: la regla de salida es |z_t| < z_salida, asi que un
# umbral de cero la vuelve inalcanzable y la posicion solo podria cerrarse por
# stop o por tenencia maxima, lo que mantiene el portafolio practicamente
# siempre invertido y desvirtua la estrategia de reversion.
# El rango de z_entrada llega a 3.0 para que el optimo no quede en la frontera
# de la malla; si el optimo se pega al extremo, la malla esta mal especificada
# y no sabemos si el verdadero optimo esta mas alla.
MALLA_Z_ENTRADA = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)
MALLA_Z_SALIDA = (0.25, 0.5, 0.75, 1.0)
MALLA_Z_STOP = (3.0, 3.5, 4.0, 4.5, 5.0)

# Periodo maximo de tenencia como multiplo de la vida media (seccion 3.6: "un
# periodo maximo de tenencia calibrado a la vida media"). Con factor 2, una
# posicion que no revirtio en dos vidas medias se cierra: la forma en que esta
# estrategia pierde dinero es una posicion que deja de revertir.
MULTIPLO_TENENCIA_MAXIMA = 2.0

# Banda de no-rebalanceo contra la que se compara el rebalanceo diario
# (seccion 3.6). Se rebalancea solo si algun peso objetivo se desvio mas que el
# ancho de la banda, medido como fraccion del nocional bruto. El ancho se elige
# por grid search SOLO con entrenamiento, igual que los umbrales del z-score, y
# se reporta el costo del rebalanceo diario contra la banda en las tres
# particiones.
MALLA_BANDA_NO_REBALANCEO = (0.01, 0.025, 0.05, 0.075, 0.10, 0.15)

# Tolerancia declarada para la igualdad de contribuciones al riesgo del Risk
# Parity (prueba obligatoria 4, nivel B).
TOLERANCIA_RISK_PARITY = 1e-6
MAX_ITERACIONES_RISK_PARITY = 10_000

# ---------------------------------------------------------------------------
# Filtros de Kalman (secciones 3.3 y 3.4)
# ---------------------------------------------------------------------------
BURN_IN_DIAS = 60  # dias habiles descartados antes de operar (~3 meses)
MALLA_DELTA_Q = (1e-6, 1e-5, 1e-4, 1e-3)  # Q = delta * Sigma_0, delta por log-verosimilitud

# Filtro 2: la ventana efectiva con la que rastrea la media y la varianza
# latentes del spread se fija como un MULTIPLO de la vida media medida en
# entrenamiento. El equilibrio tiene que moverse mas despacio que las
# desviaciones que queremos operar: si la media persiguiera al spread a la
# misma velocidad a la que este revierte, el z-score colapsaria a cero y no
# habria senal. Un factor 3 deja la reversion de ~35 dias completamente
# visible y a la vez permite que la media se reacomode ante un cambio de
# regimen real.
MULTIPLO_VENTANA_FILTRO2 = 3.0

# El filtro 2 usa varianza de observacion ADAPTATIVA, R2_t = sigma_hat^2_{t|t-1}.
# Es necesario porque se trabaja en niveles de precio y los niveles del universo
# crecieron hasta 10 veces entre 2012 y 2026: cualquier varianza constante
# calibrada en entrenamiento queda mal escalada fuera de muestra. Hacer R2
# proporcional a la varianza estimada vuelve la ganancia del filtro, y por tanto
# el z-score, invariantes a la escala del spread.
DIAS_SEMILLA_VARIANZA = 60

# ---------------------------------------------------------------------------
# Validacion de calidad de datos (seccion 3.1: ingesta, limpieza y validacion)
# ---------------------------------------------------------------------------
RETORNO_DIARIO_SOSPECHOSO = 0.50  # |r| > 50% se marca para revision manual
MAXIMO_NAN_CONSECUTIVOS = 5  # mas de una semana sin cotizar invalida el panel
