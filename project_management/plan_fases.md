# Plan por fases aprobado — Lab 03 MyST, Equipo 1

Plan aprobado por el equipo el 7 de noviembre de 2026. Es el contrato de trabajo: cada fase
tiene un entregable verificable y, al terminarla, se marca como completada en este archivo.

**Modo de trabajo (modificado el 7-nov):** no se detiene el trabajo al cerrar cada fase, sino
al cerrar cada **bloque**, con un resumen corto por fase del bloque.

| Bloque | Fases | Hito | Objetivo (hora) | Estado |
|---|---|---|---|---|
| — | F0, F1 | H1 | sáb 7-nov 14:30 | ✅ **Completado** |
| 1 | F2 → F5 | H2, H3, H4 | sáb 7-nov 21:30 | ✅ **Completado** |
| 2 | F6 → F8 | H5, H6 | dom 8-nov 11:00 | ✅ **Completado** |
| 3 | F9, F10 | H7, H8 | dom 8-nov 18:00 | ✅ **Completado** |
| 4 | Correcciones de revisión externa | — | posterior | ✅ **Completado** |

**Prioridad absoluta ante cualquier bloqueo:** (1) que `python main.py` corra completo,
(2) que `pytest` pase, (3) que el dashboard abra. Si un componente se complica, se elige la
opción más simple que cumpla el PDF y se anota como limitación en el reporte.

---

## Estado de las fases

| Fase | Entregable | Secciones del PDF | Estado |
|---|---|---|---|
| **F0** | Plan por fases, elección de universo, convenciones acordadas | 3.1, 3.2.1 | ✅ **Completada** |
| **F1** | Esqueleto del repo, `config.py` (semilla 42), ingesta/limpieza/validación, partición 60/20/20, `charter.md` | 3.1, 3.8-PM | ✅ **Completada** |
| **F2** | Screening de correlación → Engle-Granger → Johansen → validación del vector → tabla de grupos probados | 3.2 completo | ✅ **Completada** |
| **F3** | Filtro de Kalman 1 (β_t dinámico) + Johansen móvil (primer eigenvector) | 3.3 | ✅ **Completada** |
| **F4** | Filtro de Kalman 2 (μ_t, σ²_t latentes) + z-score ex-ante | 3.4 | ✅ **Completada** |
| **F5** | VECM, derivación del z-score de señal, grid search de umbrales, stop y tenencia máxima | 3.6 | ✅ **Completada** |
| **F6** | Motor de backtest multi-pata (costos, 80% consolidado, rebalanceo vs banda) + Risk Parity | 3.1-params, 3.6, 3.7 | ✅ **Completada** |
| **F7** | Métricas de desempeño, atribución de costos y de drawdown, comparativos, 9 figuras | 3.7, 3.10 | ✅ **Completada** |
| **F8** | `tests/` (4 pruebas obligatorias + nivel B) y `main.py` de un solo comando | 3.11, 3.1 | ✅ **Completada** |
| **F9** | Dashboard (vista ejecutiva + vista técnica) leyendo solo `artifacts/` | 3.8-Dashboard | ✅ **Completada** |
| **F10** | Formulaciones de Powell, ejemplo numérico, `README.md`, sprint, `postmortem.md`, `docs/reporte.pdf` | 3.5, 3.8-PM, 2.2 | ✅ **Completada** |

> Nota sobre la numeración: el PDF numera **dos** secciones como 3.8 (gestión del proyecto y
> dashboard) y no tiene 3.9. Se les llama 3.8-PM y 3.8-Dashboard.

---

## Arquitectura de módulos

```
main.py                        config.py                 artifacts_io.py
data/      universe.py loader.py cleaning.py validation.py splits.py
cointegration/  screening.py engle_granger.py johansen.py spread.py selection.py rolling.py
kalman/    base.py filter1_weights.py filter2_signal.py calibration.py worked_example.py
strategy/  vecm.py zscore.py signals.py sizing.py thresholds.py
backtest/  engine.py costs.py rebalance.py risk_parity.py walkforward.py
performance/ metrics.py drawdown.py attribution.py report_figures.py
dashboard/ app.py data_access.py views_executive.py views_technical.py
tests/     conftest.py test_causality.py test_normalization.py test_accounting.py
           test_filter.py test_risk_parity.py
```

Desviación aprobada frente al plan original: el escritor de artifacts es `artifacts_io.py`
en la raíz, no `backtest/artifacts_io.py`, junto con `config.py`. Justificación en el README.

---

## Checklist de requisitos

### 3.1 Datos y parámetros

| Requisito | Dónde | Estado |
|---|---|---|
| Ventana 10–15 años, diarios | `data/loader.py`, `config.py` | ✅ 13.90 años |
| Grupos de 4 activos = 2·n | `data/universe.py`, `config.py` | ✅ |
| División 60/20/20 cronológica | `data/splits.py` + `tests/test_causality.py` | ✅ 2097/699/700 |
| Comisión 0.125% por pata, entrada y salida | `backtest/costs.py` | ✅ |
| Préstamo 0.25% anual devengo diario | `backtest/costs.py` | ✅ actual/365 |
| Capital inicial $1,000,000 | `config.py` | ✅ |
| Posición 80% del capital consolidado | `strategy/sizing.py`, `backtest/walkforward.py` | ✅ verificado en prueba 3 |
| Semilla 42 documentada | `config.py::SEMILLA`, `README.md` | ✅ |
| `requirements.txt` con versiones fijadas | raíz | ✅ |

### 3.2 Selección del grupo

| Paso | Dónde | Estado |
|---|---|---|
| 1. Universo justificado | `data/universe.py`, `README.md`, reporte | ✅ |
| 2. Correlación móvil > 0.7 | `cointegration/screening.py` | ✅ |
| 3. Pre-filtro Engle-Granger (OLS + ADF, p<0.05) | `cointegration/engle_granger.py` | ✅ |
| 4. Johansen: traza, max-eigen, rango, vector más reversivo | `cointegration/johansen.py` | ✅ |
| 5. ADF del spread + vida media | `cointegration/spread.py` | ✅ |
| 6. Tabla de grupos probados + etapa de descarte + total `m` | `cointegration/selection.py` → `artifacts/selection/groups_tested.csv` | ✅ m=26 |
| Tabla comparativa de columnas cointegrantes | `artifacts/selection/vector_comparison_{grupo}.csv` | ✅ |

### 3.3 / 3.4 Filtros de Kalman

| Requisito | Dónde | Estado |
|---|---|---|
| Estado β_t, observación P_t, salida S_t | `kalman/filter1_weights.py` | ✅ |
| Normalización y signo declarados | `cointegration/spread.py` + `tests/test_normalization.py` | ✅ |
| Inicialización, Q, R, burn-in justificados | `kalman/calibration.py`, reporte | ✅ |
| Estado (μ_t, σ²_t), z-score **pre-update** | `kalman/filter2_signal.py` | ✅ |
| Ganancia de Kalman registrada | `kalman/base.py` → artifacts | ✅ |

### 3.5 Marco de Powell

| Requisito | Dónde | Estado |
|---|---|---|
| Proceso de 6 pasos + 5 elementos, filtro 1 | `docs/reporte.pdf` §6.1 | ✅ |
| Idem, filtro 2 | `docs/reporte.pdf` §6.2 | ✅ |
| Idem, sistema completo con componentes física / de información / de creencia | `docs/reporte.pdf` §6.3 | ✅ |
| Ganancia de Kalman: cálculo e interpretación | `docs/reporte.pdf` §6.4 | ✅ |
| Ejemplo numérico trabajado de varios periodos | `kalman/worked_example.py` → `artifacts/kalman/worked_example_*.csv`, reporte §6.5 | ✅ |

### 3.6 VECM y señales

| Requisito | Dónde | Estado |
|---|---|---|
| VECM con ECT de Johansen | `strategy/vecm.py` | ✅ |
| z-score de residuales VECM normalizado con filtro 2 | `strategy/zscore.py` | ✅ |
| Umbrales + grid search documentado (solo train) | `strategy/thresholds.py` → `artifacts/strategy/grid_search_{grupo}.csv` | ✅ 171 combinaciones |
| Entradas/salidas long/short del spread | `strategy/signals.py` | ✅ |
| Tamaño proporcional a β̂_t con signo | `strategy/sizing.py` | ✅ |
| Rebalanceo diario + costo vs banda de no-rebalanceo | `backtest/rebalance.py` → `artifacts/backtest/rebalance_comparison.csv` | ✅ ancho calibrado en train, comparado en 3 particiones |
| Stop sobre \|z_t\| + tenencia máxima calibrada a vida media | `strategy/signals.py` | ✅ |

### 3.7 Risk Parity (nivel B)

| Requisito | Dónde | Estado |
|---|---|---|
| Proceso completo independiente por grupo | `backtest/portfolio.py` | ✅ |
| Grupos disjuntos, justificados por separado | `data/universe.py`, reporte | ✅ |
| RP sobre retornos de spreads | `backtest/risk_parity.py` | ✅ |
| Verificación numérica de contribuciones iguales | `tests/test_risk_parity.py` | ✅ desvío 0.0e+00 |
| Matriz de correlación entre spreads + discusión | `artifacts/riskparity/spread_corr.csv`, reporte | ✅ −0.1374 |
| Consolidado vs cada grupo vs pesos iguales | `performance/metrics.py` → `artifacts/performance/metrics.json` | ✅ |
| Atribución del max drawdown por grupo | `performance/drawdown.py` | ✅ B1 87.6% |
| 80% aplica al consolidado | `backtest/walkforward.py` | ✅ |

### 3.8-PM Gestión del proyecto

| Requisito | Dónde | Estado |
|---|---|---|
| Charter: alcance, hitos, ≥5 riesgos, roles | `project_management/charter.md` | ✅ 8 riesgos |
| Sprints con objetivo, tareas y bloqueos | `project_management/sprint_01.md` | ✅ |
| Post-mortem de 1–2 páginas | `project_management/postmortem.md` | ✅ borrador con [REVISAR] |

### 3.8-Dashboard

| Vista | Elemento | Estado |
|---|---|---|
| Ejecutiva | Curva de capital con drawdown superpuesto | ✅ |
| Ejecutiva | Tarjeta: Sharpe, Sortino, Calmar, MaxDD, Win Rate, Profit Factor | ✅ |
| Ejecutiva | Atribución de costos: bruto → neto, comisiones, préstamo | ✅ |
| Ejecutiva | Bitácora filtrable por periodo, dirección y resultado | ✅ |
| Ejecutiva | Histograma de retornos por operación con normal superpuesta | ✅ |
| Técnica | Spread con señales anotadas | ✅ |
| Técnica | Componentes de β̂_t, una línea por activo | ✅ |
| Técnica | Primer eigenvector de Johansen móvil | ✅ |
| Técnica | Ganancia de Kalman de ambos filtros | ✅ |
| Técnica | z-score con líneas de umbral | ✅ |
| Técnica | Varianza posterior (convergencia de los filtros) | ✅ |
| Técnica | Sharpe móvil de 252 días | ✅ |
| Requisito | Un solo comando, gráficas interactivas, solo lee `artifacts/` | ✅ |

### 3.10 Nueve figuras obligatorias

| # | Figura | Generador | Artifact fuente | Estado |
|---|---|---|---|---|
| 1 | Precios individuales | `report_figures.py::fig01_prices` | `prices.parquet` | ✅ |
| 2 | Spread con señales | `fig02_spread_signals` | `strategy/signals_{g}.parquet` | ✅ |
| 3 | Vector de pesos en el tiempo | `fig03_beta_paths` | `kalman/filter1_{g}.parquet` | ✅ |
| 4 | Primer eigenvector en el tiempo | `fig04_rolling_eigvec` | `rolling/johansen_eigen_{g}.parquet` | ✅ |
| 5 | Ganancia de Kalman, ambos filtros | `fig05_kalman_gains` | `kalman/filter1_{g}` + `filter2_{g}` | ✅ |
| 6 | z-score con umbrales | `fig06_zscore` | `strategy/signals_{g}.parquet` | ✅ |
| 7 | Curvas de capital train/test/validación | `fig07_equity_splits` | `backtest/portfolio_equity.parquet` | ✅ |
| 8 | Distribución de retornos por operación | `fig08_trade_returns` | `backtest/trades_all.csv` | ✅ |
| 9 | Contribuciones al riesgo + corr. entre spreads | `fig09_risk_parity` | `riskparity/weights.csv`, `spread_corr.csv` | ✅ |

### 3.11 Pruebas obligatorias

| # | Prueba | Archivo | Aserción | Estado |
|---|---|---|---|---|
| 1 | Causalidad | `tests/test_causality.py` | Para 3 fechas t, el pipeline truncado en t reproduce z_t con tolerancia 1e-10 | ✅ |
| 2 | Normalización | `tests/test_normalization.py` | β̂_{i*,t}=1 ∀t; todo cambio de signo es cruce continuo por cero | ✅ |
| 3 | Contabilidad multi-pata | `tests/test_accounting.py` | V_t = efectivo + Σ patas; costo = #ops×comisión + devengo | ✅ |
| 4 | Filtro | `tests/test_filter.py` | Var ∈ [0.8,1.25]; autocorrelación de AMBOS filtros reportada y contrastada contra φ=0.5^(1/h); Ljung-Box del VECM p>0.05 | ✅ |
| 4-B | Risk parity (nivel B) | `tests/test_risk_parity.py` | max_g \|RC_g − 1/G\| < 1e-6 | ✅ |

---

## Convenciones acordadas

Estas convenciones fueron discutidas y aprobadas antes de escribir código. Van completas en
el reporte porque el equipo debe poder defenderlas.

### C1. Ecuación de observación del filtro 1

Pata numeraria i\* fija, con el resto de precios en la matriz de observación. Sea
$\tilde P_t=(P_{j,t})_{j\neq i^*}$ y el estado $x_t=(\theta_t^\top, c_t)^\top$:

$$x_t = x_{t-1} + w_t,\qquad w_t\sim\mathcal N(0,Q)$$
$$y_t = P_{i^*,t} = H_t x_t + v_t,\qquad H_t=\begin{pmatrix}\tilde P_t^\top & 1\end{pmatrix},\qquad v_t\sim\mathcal N(0,R)$$

El vector se reconstruye con $\hat\beta_{i^*,t}=1$ y $\hat\beta_{j,t}=-\hat\theta_{j,t}$, y el
spread operable es la **innovación**, es decir el estado predicho y nunca el posterior:

$$S_t = \hat\beta_{t|t-1}^\top P_t - \hat c_{t|t-1} = \nu_t$$

Decisiones declaradas: precios en **niveles** (no logaritmos) para que β se traduzca a número
de acciones y la contabilidad cierre; **intercepto** $c_t$ en el estado (equivalente a
constante en la relación de cointegración); $R=\widehat{\mathrm{Var}}(\hat u^{\text{train}})$;
$Q=\delta\Sigma_0$ con $\Sigma_0$ la covarianza OLS de $\hat\theta^{\text{train}}$ y $\delta$
elegido en malla por log-verosimilitud **solo en entrenamiento**; burn-in de 60 días hábiles.

### C2. Normalización y signo de β

$$i^* = \arg\max_i \lvert \beta^{J}_i\rvert, \qquad \beta \leftarrow \frac{\beta^{J}}{\beta^{J}_{i^*}}$$

Dividir por la componente **con su signo** fija de un golpe la escala y la indeterminación de
signo del eigenvector, dejando $\beta_{i^*}=+1$. La pata numeraria se elige **una sola vez**
en entrenamiento y queda congelada; re-elegirla por ventana rompería la prueba 2. "Largo el
spread" significa siempre largo 1 unidad de la pata numeraria. Para la figura 4, cada ventana
del Johansen móvil se re-ancla con la misma regla antes de graficar.

### C3. Devengo del préstamo: actual/365 sobre días calendario

$$\text{costo}_t = 0.0025 \times \frac{\Delta_t^{\text{cal}}}{365}\times \sum_{j:\,q_{j,t}<0}\lvert q_{j,t}\rvert P_{j,t}$$

Se cobra fin de semana y feriados, porque un corto los paga, y con nocional constante el costo
de un año completo es exactamente 0.25%. **365 es solo para el devengo**: la anualización de
Sharpe, Sortino y el Sharpe móvil usa **252** días hábiles.

### C4. Archivos de `artifacts/`

Series de tiempo en Parquet, tablas en CSV, escalares y metadatos en JSON. El dashboard solo
lee de aquí.

```
artifacts/
├── manifest.json
├── prices.parquet
├── selection/   groups_tested.csv  johansen_{g}.json  vector_comparison_{g}.csv
│                particiones.csv  calidad_datos.csv
├── kalman/      filter1_{g}.parquet  filter2_{g}.parquet  worked_example_{g}.csv
├── rolling/     johansen_eigen_{g}.parquet
├── strategy/    signals_{g}.parquet  grid_search_{g}.csv
├── backtest/    portfolio_equity.parquet  group_equity_{g}.parquet
│                equity_equal_weight.parquet  trades_all.csv  trades_{g}.csv
│                legs_ledger_{g}.parquet  rebalance_comparison.csv
├── riskparity/  weights.csv  spread_returns.parquet  spread_corr.csv
└── performance/ metrics.json  rolling_sharpe.parquet  cost_attribution.json
                 drawdown_attribution.csv
```

### C5. De dónde sale el z-score de señal

El filtro 2 observa $S_t$, que **es** el término de corrección de error $\beta^\top P_t$, y el
VECM aporta el coeficiente de ajuste $\alpha$, su significancia y la vida media implícita
$h=-\ln 2/\ln(1+\alpha_{i^*})$ que calibra la tenencia máxima. Así $z_t$ conserva la
interpretación de desviación del equilibrio, que es lo que la regla de trading necesita. El
residuo de pronóstico a un paso del VECM se reporta como diagnóstico, no como insumo de la
señal. **Se documenta en el reporte que esta es nuestra interpretación de "residuales del
VECM"** y por qué es consistente con la sección 3.4: la sección 3.4 exige que el filtro 2
observe el spread realizado y produzca el z-score con estadísticos previos a la actualización,
y el ECT es precisamente el spread realizado.

### C6. Qué significa el 80%

Nocional **bruto** (suma de valores absolutos de todas las patas). Con $C_t$ el capital
consolidado y $w_g$ el peso de Risk Parity:

$$u_{g,t} = \frac{0.80\,C_t\,w_g}{\sum_k \lvert\hat\beta_{k,t}\rvert P_{k,t}},\qquad q_{j,t} = d_{g,t}\,u_{g,t}\,\hat\beta_{j,t},\quad d_{g,t}\in\{-1,0,+1\}$$

El retorno del spread que alimenta Risk Parity se define por unidad de nocional bruto,
$r_{g,t}=\Delta S_{g,t}/\sum_k\lvert\hat\beta_{k,t-1}\rvert P_{k,t-1}$, para que la covarianza
entre grupos sea comparable.

### C7. Defensa frente a pruebas múltiples

Se reporta el número total de grupos evaluados $m$ y el número esperado de falsos positivos
$\alpha\,m$ bajo la hipótesis nula, con $\alpha=0.05$. Con los dos pools primarios,
$m=\binom{6}{4}+\binom{5}{4}=15+5=20$ y $\alpha m = 1$: **esperamos un grupo espurio y lo
decimos**. Sin restricción económica, buscar grupos de 4 en el S&P 500 daría
$\binom{500}{4}\approx 2.57\times10^{9}$ hipótesis y $\alpha m \approx 1.3\times 10^{8}$
falsos positivos esperados. No se aplican correcciones de multiplicidad tipo Bonferroni por
decisión del equipo: no se vieron en clase y el curso pide ceñirse a lo visto.

### C8. Universo evaluado y grupos operados

Universo original A, cadena de valor del crudo: **A1** = E&P/integradas (XOM, CVX, COP, OXY,
EOG, DVN), tendencia comun el nivel del crudo; **A2** = refinacion (VLO, MPC, PSX, PBF, DINO),
tendencia comun el margen de refinacion; **A3** = servicios (SLB, HAL, NOV, RIG) como reserva.

Resultado del embudo: ningun grupo de A1 paso Johansen al 95%, lo que activo A3 por la regla
declarada; A3 tampoco paso. Para poder construir el portafolio de Risk Parity de nivel B, que
exige al menos dos grupos cointegrados disjuntos, se registro **despues** el pool **B1** =
bancos canadienses listados en NYSE (RY, TD, BNS, BMO, CM), evaluado con exactamente los
mismos criterios y umbrales. No se relajo ningun criterio ni se probaron especificaciones
adicionales para que A1 o A3 pasaran.

Correccion a la justificacion inicial de B1: son **cinco bancos distintos**, no el mismo
instrumento en dos bolsas. Cada uno esta cross-listed individualmente en NYSE y TSX, pero
entre ellos no comparten flujos de efectivo. Lo que comparten es el entorno: el mismo
regulador prudencial (OSFI), el mismo mercado hipotecario domestico con seguro publico
(CMHC) y el mismo ciclo de tasas del Banco de Canada.

**Grupos operados:** A2 = VLO-MPC-PSX-PBF (numeraria VLO) y B1 = RY-TD-BMO-CM (numeraria CM).
Disjuntos, verificado por codigo. `m` = 26 grupos evaluados, `alpha*m` = 1.3 falsos positivos
esperados, 2 grupos pasaron el embudo completo.
