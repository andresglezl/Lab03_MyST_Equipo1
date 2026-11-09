# Project Charter — Lab 03 MyST, Equipo 1

**Proyecto:** Arbitraje estadístico con cointegración y filtros de Kalman
**Nivel de alcance:** B (Risk Parity entre varios grupos cointegrados)
**Integrantes:** Rogelio Adrián Arroyo Valencia · Ernesto Andrés González Lomelí

| | |
|---|---|
| Arranque del proyecto | Sábado 7 de noviembre de 2026, ~14:30 |
| **Límite de entrega del repositorio** | **Domingo 8 de noviembre de 2026, 23:59** |
| **Exposición** | **Lunes 9 de noviembre de 2026** |
| Tiempo total disponible | ~33 horas de calendario (**un solo sprint**) |
| Congelamiento de código objetivo | Domingo 8 de noviembre, 18:00 (margen de 6 h antes del límite) |

El proyecto se ejecuta en **un único sprint** ([`sprint_01.md`](sprint_01.md)) porque el
tiempo total es menor a 48 horas. Los hitos se calendarizan por hora, no por semana, y el
trabajo se organiza en tres bloques de fases con un punto de revisión al cierre de cada uno.

---

## 1. Declaración de alcance

### Qué hará la estrategia

- Pre-registrar un universo de candidatos justificado económicamente (cadena de valor del
  crudo: E&P/integradas y refinación) y probar **todos** sus subconjuntos de 4 activos.
- Seleccionar grupos cointegrados con el embudo correlación móvil → Engle-Granger → Johansen
  → validación del spread (ADF + vida media), **estimado solo con la partición de
  entrenamiento**.
- Rastrear el vector de pesos cointegrante con un filtro de Kalman (filtro 1) y la media y
  varianza latentes del spread con un segundo filtro (filtro 2), generando un z-score sin
  look-ahead.
- Estimar un VECM con el vector de Johansen como término de corrección de error y operar
  entradas, salidas, stop y tenencia máxima sobre el z-score, con umbrales calibrados por
  grid search en entrenamiento.
- Operar **dos grupos disjuntos simultáneamente** con capital asignado por Risk Parity sobre
  los retornos de los spreads, verificando numéricamente la igualdad de contribuciones al
  riesgo.
- Simular con costos reales: comisión 0.125% por pata a la entrada y a la salida, y préstamo
  0.25% anual actual/365 sobre el nocional en corto.
- Entregar un dashboard con vista ejecutiva y vista técnica que lee únicamente `artifacts/`.

### Qué NO hará

- No usa aprendizaje automático, modelos GARCH, optimizadores de cartera sofisticados ni
  ninguna técnica fuera de los lineamientos y de lo visto en clase.
- No opera intradía: una decisión por día sobre precios de cierre ajustados.
- No modela impacto de mercado, slippage, disponibilidad de préstamo ni límites de
  concentración del bróker. El costo es comisión + préstamo, nada más.
- No incluye activos deslistados o adquiridos, lo que introduce un **sesgo de supervivencia**
  en el universo que se declara como limitación, no se corrige.
- No afirma que la relación de equilibrio encontrada sea estructural: la sostiene o la
  descarta con el mecanismo económico declarado y con el desempeño fuera de muestra.

---

## 2. Plan de hitos

Calendarizado por hora. "Bloque" es el punto de revisión: no se avanza al siguiente bloque
sin que el anterior corra.

| # | Hito | Entregable verificable | Bloque | Fecha y hora objetivo | Estado |
|---|---|---|---|---|---|
| H1 | Pipeline de datos completo | `python main.py` produce `artifacts/prices.parquet` validado y particionado 60/20/20 | F1 | sáb 7-nov 14:30 | ✅ Completado |
| H2 | Prueba de cointegración completa | `artifacts/selection/groups_tested.csv` con los `m` grupos probados, su etapa de descarte y los grupos seleccionados | 1 (F2) | sáb 7-nov 17:00 | ⬜ |
| H3 | Filtro 1 validado | `artifacts/kalman/filter1_{grupo}.parquet` con β̂ₜ, ganancia e innovaciones normalizadas de varianza ≈ 1 | 1 (F3) | sáb 7-nov 19:00 | ⬜ |
| H4 | Filtro 2, VECM y señales | z-score ex-ante, α del VECM y umbrales calibrados por grid search en entrenamiento | 1 (F4–F5) | sáb 7-nov 21:30 | ⬜ |
| H5 | Backtest completo y Risk Parity | Curvas de capital de los tres tramos, bitácora de operaciones y contribuciones al riesgo verificadas | 2 (F6) | dom 8-nov 02:00 | ⬜ |
| H6 | Métricas, 9 figuras y pruebas en verde | `pytest -q` pasa las 4 pruebas obligatorias + nivel B | 2 (F7–F8) | dom 8-nov 11:00 | ⬜ |
| H7 | Dashboard funcionando | `streamlit run dashboard/app.py` con las dos vistas, leyendo solo `artifacts/` | 3 (F9) | dom 8-nov 14:00 | ⬜ |
| H8 | README, Powell y reporte | `docs/reporte.pdf` con las tres formulaciones y el ejemplo numérico; guía del dashboard | 3 (F10) | dom 8-nov 18:00 | ⬜ |
| — | **Entrega del repositorio** | Último commit antes del límite | — | **dom 8-nov 23:59** | ⬜ |

### Regla de priorización bajo presión de tiempo

Con ~33 horas y sin margen para iterar, el orden de prioridad es explícito y se aplica ante
cualquier bloqueo: **(1)** que `python main.py` corra completo, **(2)** que `pytest` pase,
**(3)** que el dashboard abra. Si un componente se complica, se elige la opción **más simple
que cumpla el PDF** y la simplificación se registra como limitación en el reporte, en lugar
de gastar tiempo en refinarla. Un entregable completo y honesto sobre sus límites puntúa más
alto en la rúbrica que uno sofisticado e incompleto.

---

## 3. Registro de riesgos

Probabilidad e impacto en escala Alta / Media / Baja.

| # | Riesgo | Prob. | Impacto | Estrategia de mitigación |
|---|---|---|---|---|
| R1 | **Pérdida de cointegración fuera de muestra.** El grupo coselecciona en entrenamiento y la relación se rompe en prueba o validación. | Alta | Alto | Exigir mecanismo económico nombrado por grupo, no solo p-value. Reportar ADF y vida media por partición. El stop sobre \|z\| y la tenencia máxima calibrada a la vida media acotan la pérdida de una posición que deja de revertir. |
| R2 | **Hallazgo espurio por pruebas múltiples.** Con `m` grupos probados a α = 0.05 se esperan α·m falsos positivos. | Alta | Alto | Universo pre-registrado y congelado en `data/universe.py` antes de ver estadísticos. Reportar `m` total y el número esperado de falsos positivos α·m en el reporte y en la exposición. |
| R3 | **Divergencia o sobre-reacción del filtro de Kalman.** Un Q demasiado grande hace que β̂ₜ persiga ruido; uno demasiado chico lo congela. | Media | Alto | Q = δ·Σ₀ con un único escalar δ elegido en malla por log-verosimilitud **solo en entrenamiento**. Burn-in de 60 días. Diagnóstico obligatorio: varianza de innovaciones normalizadas ≈ 1 y Ljung-Box no significativa (prueba 4). |
| R4 | **Calidad de datos.** Splits sin ajustar, huecos, ticker renombrado (HFC → DINO) o precio erróneo de la fuente. | Media | Alto | Cierre ajustado por splits y dividendos; validación automática que separa fallas fatales de observaciones y detiene el pipeline ante las fatales. Cache versionado del panel crudo. |
| R5 | **Diversificación aparente entre grupos.** Los dos spreads comparten factores macro y el Risk Parity reparte riesgo que en realidad es el mismo. | Media | Medio | Elegir grupos en lados opuestos del mismo insumo (A1 ingreso / A2 costo del crudo). Reportar la matriz de correlación entre spreads y atribuir el drawdown máximo del consolidado a los grupos que lo originaron. |
| R6 | **Los costos se comen el edge.** El rebalanceo diario de β̂ₜ genera turnover y 0.125% por pata lo cobra todo. | Alta | Medio | Cuantificar el costo del rebalanceo diario contra una política de banda de no-rebalanceo y reportar ambas. Atribución explícita de P&L bruto → neto, comisiones y préstamo. |
| R7 | **Look-ahead accidental.** Normalizar con la media posterior del filtro 2 o reestimar con datos futuros. | Media | Alto | z-score calculado con estadísticos **previos** a la actualización. Prueba de causalidad automatizada: recalcular el pipeline truncado en t reproduce z_t para al menos tres valores de t. |
| R8 | **Riesgo de calendario.** ~33 horas totales entre el arranque y el límite de entrega, con fases estrictamente secuenciales (sin cointegración no hay filtro, sin filtro no hay señal). Un bloqueo en una fase se propaga a todas. | Alta | Alto | Hitos por hora con criterio de salida verificable y revisión al cierre de cada bloque. Regla de priorización declarada arriba: simplificar y documentar la limitación antes que refinar. Congelamiento de código a las 18:00 del domingo, 6 h antes del límite. |

---

## 4. Asignación de roles

Las contribuciones esperadas en Git se listan por archivo para que el historial de commits
sea verificable contra este charter (la rúbrica penaliza con 0 en la parte técnica a un
integrante sin commits propios).

### Rogelio Adrián Arroyo Valencia — Datos, cointegración y señales

- `data/` completo: `universe.py`, `loader.py`, `cleaning.py`, `validation.py`, `splits.py`
- `cointegration/` completo: `screening.py`, `engle_granger.py`, `johansen.py`, `spread.py`,
  `selection.py`, `rolling.py`
- `strategy/vecm.py`, `strategy/zscore.py`, `strategy/signals.py`, `strategy/thresholds.py`
- `tests/test_causality.py`, `tests/test_normalization.py`
- Secciones del reporte: 3.2 (selección y defensa frente a pruebas múltiples), 3.6 (VECM y
  señales), formulación de Powell del **filtro 2**

### Ernesto Andrés González Lomelí — Filtros, backtest y dashboard

- `kalman/` completo: `base.py`, `filter1_weights.py`, `filter2_signal.py`,
  `calibration.py`, `worked_example.py`
- `backtest/` completo: `engine.py`, `costs.py`, `rebalance.py`, `risk_parity.py`,
  `walkforward.py`
- `performance/` completo y `strategy/sizing.py`
- `dashboard/` completo
- `tests/test_accounting.py`, `tests/test_filter.py`, `tests/test_risk_parity.py`
- Secciones del reporte: 3.3 y 3.4 (filtros), 3.7 (Risk Parity), 3.10 (figuras),
  formulación de Powell del **filtro 1**

### Compartido

- `config.py`, `artifacts_io.py`, `main.py`, `README.md`, `requirements.txt`
- Formulación de Powell del **sistema completo** (ambos filtros + VECM + lógica de trading)
- `project_management/` y `docs/reporte.pdf`

Regla de trabajo: cada integrante debe poder explicar cualquier línea del proyecto, no solo
la de su rol. Antes de la exposición se hace una sesión cruzada en la que cada uno explica
los módulos del otro.
