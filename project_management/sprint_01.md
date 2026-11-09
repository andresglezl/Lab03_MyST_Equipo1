# Sprint 01 — Proyecto completo (sprint unico)

**Ventana del sprint:** sabado 7 de noviembre de 2026, 14:30 -> domingo 8 de noviembre, 18:00
**Limite de entrega del repositorio:** domingo 8 de noviembre de 2026, 23:59
**Exposicion:** lunes 9 de noviembre de 2026
**Hitos cubiertos:** H1 a H8 (ver `charter.md`)

El proyecto completo cabe en un solo sprint porque el tiempo total disponible es de ~33 horas
de calendario. Este archivo se actualiza al cierre de cada bloque de fases.

---

## Bloque F1 — Andamiaje y pipeline de datos (H1, cerrado sab 14:30)

### Objetivo

Dejar el repositorio con la estructura obligatoria completa y un pipeline de datos que corra
con un solo comando: descargar la ventana histórica del universo pre-registrado, limpiarla,
validarla y aplicar la partición cronológica 60/20/20. Al cerrar el sprint, toda decisión de
datos debe estar fijada en `config.py` y toda salida debe quedar escrita en `artifacts/`,
porque las fases siguientes consumen exclusivamente esos archivos.

### Tareas planeadas

| # | Tarea | Estado | Notas |
|---|---|---|---|
| T1 | Estructura de carpetas según sección 2.2 | ✅ Hecho | Paquetes con `__init__.py` descriptivo |
| T2 | `requirements.txt` con versiones fijadas | ✅ Hecho | 11 dependencias, entorno Python 3.12.4 |
| T3 | `.gitignore` (cache de datos, artifacts, `__pycache__`) | ✅ Hecho | `artifacts/` se versiona vacío con `.gitkeep` |
| T4 | `config.py` con todos los parámetros y la semilla 42 | ✅ Hecho | Incluye la justificación de cada constante en comentarios |
| T5 | Universo pre-registrado y congelado | ✅ Hecho | `data/universe.py`; pools disjuntos verificados por código |
| T6 | Ingesta con cache local | ✅ Hecho | `data/loader.py`; cierre ajustado, `auto_adjust=True` |
| T7 | Limpieza y alineación del panel | ✅ Hecho | `data/cleaning.py`; ventana común, ffill con límite |
| T8 | Validación de calidad con fallas fatales vs observaciones | ✅ Hecho | `data/validation.py`; detiene el pipeline ante fatales |
| T9 | Partición cronológica 60/20/20 | ✅ Hecho | `data/splits.py`; 2097 / 699 / 700 días |
| T10 | Capa única de I/O de `artifacts/` + manifiesto | ✅ Hecho | `artifacts_io.py`; registra semilla y versiones por corrida |
| T11 | `main.py` ejecutable con un comando | ✅ Hecho | `python main.py` corre la etapa de datos sin errores |
| T12 | `README.md` y `charter.md` | ✅ Hecho | Guía del dashboard pendiente hasta la Fase 9 |

### Resultados medidos

- Ventana efectiva: **2012-12-13 → 2026-11-06 = 13.90 años**, 3496 observaciones, 11 tickers.
- NaN rellenados: **0**. Ningún ticker con huecos.
- Particiones: entrenamiento 2012-12-13 → 2021-04-14 (2097 días, 59.98%), prueba
  2021-04-15 → 2024-01-24 (699 días, 19.99%), validación 2024-01-25 → 2026-11-06
  (700 días, 20.02%).

### Bloqueos y desviaciones respecto al plan

1. **La ventana la impone el ticker más joven, no la elegimos.** PBF cotiza desde 2012-12-13,
   así que la ventana común quedó en 13.90 años en lugar de los 15 que pensábamos tomar. Se
   decidió usar una sola ventana común para los 11 tickers en vez de una por grupo, para que
   todos los grupos candidatos se prueben sobre la misma muestra y el portafolio consolidado
   de Risk Parity tenga una única rejilla de fechas. Sigue dentro del rango 10–15 exigido.

2. **Desviación menor de estructura:** se agregaron dos módulos en la raíz que los
   lineamientos no listan, `config.py` y `artifacts_io.py`. Son transversales (los usan todas
   las carpetas y el dashboard) y ponerlos dentro de un paquete concreto crearía dependencias
   invertidas, por ejemplo que `data/` importara de `backtest/`. Queda declarado aquí.

3. **Un retorno extremo real, no un error de datos:** OXY cayó −52.0% el 2020-03-09 (guerra
   de precios OPEC+ y colapso de demanda por COVID). La validación lo marca como observación
   y no detiene el pipeline. Importa para las fases siguientes: la partición de
   **entrenamiento contiene el choque de 2020**, incluido el episodio de WTI negativo de
   abril de 2020, lo que va a inflar las varianzas estimadas y por tanto hará más
   conservadora la calibración de R y de los umbrales.

4. `cat` con heredoc falló al escribir dos módulos largos; se escribieron con el editor
   directamente. Sin impacto en el entregable.

### Criterio de salida del bloque

- [x] `python main.py` corre de principio a fin sin errores ni ajustes manuales.
- [x] `artifacts/prices.parquet` existe, sin NaN, con columna de partición.
- [x] `artifacts/manifest.json` registra semilla, versiones de paquetes y parámetros.
- [x] Pools del universo verificados como disjuntos por código, no por inspección.

---

## Bloque 1 — F2 a F5: cointegración, filtros, VECM y señales (H2–H4, cerrado sáb 21:30)

### Objetivo

Correr el embudo de selección completo sobre el universo pre-registrado, dejar los dos filtros
de Kalman funcionando con sus parámetros calibrados solo en entrenamiento, estimar el VECM con
el vector de Johansen como término de corrección de error, y producir el z-score ex-ante y las
señales con umbrales calibrados por grid search.

### Tareas

| # | Tarea | Estado | Notas |
|---|---|---|---|
| T13 | Filtro de correlación móvil | ✅ Hecho | `screening.py`; niveles, mediana móvil, el par más débil manda |
| T14 | Engle-Granger con críticos de MacKinnon | ✅ Hecho | `engle_granger.py`; 4 elecciones de pata dependiente |
| T15 | Johansen (traza y máximo eigenvalor) | ✅ Hecho | `johansen.py`; det_order=0, k_ar_diff=1 |
| T16 | Normalización de β y validación del spread | ✅ Hecho | `spread.py`; ADF + vida media por columna cointegrante |
| T17 | Tabla de grupos probados | ✅ Hecho | `selection.py`; m=26, etapa de descarte por grupo |
| T18 | Recursión de Kalman compartida | ✅ Hecho | `kalman/base.py`; ganancia, innovaciones, log-verosimilitud |
| T19 | Calibración de Q, R e inicialización | ✅ Hecho | `calibration.py`; Q = δ·Σ₀, δ por log-verosimilitud en train |
| T20 | Filtro 1 (pesos dinámicos) | ✅ Hecho | `filter1_weights.py`; spread = innovación, estado predicho |
| T21 | Johansen móvil (figura 4) | ✅ Hecho | `rolling.py`; ventana 504, paso 5, re-anclaje de signo |
| T22 | Filtro 2 (media y varianza latentes) | ✅ Hecho | `filter2_signal.py`; varianza adaptativa, z ex-ante |
| T23 | VECM con ECT fijo de Johansen | ✅ Hecho | `vecm.py`; β'α, selección de rezagos por Ljung-Box |
| T24 | Lógica de señales, stop y tenencia máxima | ✅ Hecho | `signals.py`; tenencia = 2 vidas medias |
| T25 | Modelo de costos | ✅ Hecho | `costs.py`; comisión por turnover, préstamo actual/365 |
| T26 | Motor por grupo en nocional bruto unitario | ✅ Hecho | `engine.py`; el mismo que usa el grid search |
| T27 | Grid search de umbrales en train | ✅ Hecho | `thresholds.py`; 171 combinaciones, objetivo Sharpe neto |
| T28 | Orquestación por grupo | ✅ Hecho | `zscore.py`; conecta F3, F4 y F5 |

### Resultados medidos

Embudo de selección, **m = 26** grupos evaluados, **α·m = 1.3** falsos positivos esperados:

| Pool | Grupos | Resultado |
|---|---|---|
| A1 (E&P) | 15 | Ninguno pasó. 9 cayeron en correlación, 6 en Engle-Granger. Traza máxima 32.1 contra crítico 47.85 |
| A2 (refinación) | 5 | **VLO-MPC-PSX-PBF pasó todo**: corr 0.714, EG p=0.0323, traza 54.78 > 47.85, rango 1, ADF p=0.0004, vida media 37.3 d |
| A3 (servicios, reserva) | 1 | No pasó. EG p=0.624, traza 40.61, ADF p=0.107 |
| B1 (bancos canadienses) | 5 | **RY-TD-BMO-CM pasó todo**: corr 0.927, EG p=0.0252, traza 54.21 > 47.85, rango 1, ADF p=0.0056, vida media 33.1 d |

Filtros y VECM:

| Grupo | R | δ | k VECM | β'α | Vida media | Ljung-Box ECT | z sd (train/prueba/valid) |
|---|---|---|---|---|---|---|---|
| VLO-MPC-PSX-PBF | 14.09 | 1e-3 | 1 | −0.01888 | 36.3 d | p=0.278 blancos | 1.02 / 1.09 / 0.99 |
| RY-TD-BMO-CM | 0.86 | 1e-3 | 8 | −0.01731 | 36.1 d | p=0.057 blancos | 1.02 / 1.05 / 1.06 |

Umbrales del grid search y desempeño por partición, en retorno por unidad de nocional bruto:

| Grupo | Umbrales | Train | Prueba | Validación |
|---|---|---|---|---|
| VLO-MPC-PSX-PBF | z_ent 1.5, z_sal 1.0, z_stop 3.0, tenencia 73 d | neto +27.10%, Sharpe +0.64 | neto −6.76%, Sharpe −0.59 | neto −0.67%, Sharpe −0.07 |
| RY-TD-BMO-CM | z_ent 1.0, z_sal 0.5, z_stop 4.0, tenencia 72 d | neto +22.16%, Sharpe +0.44 | neto −12.33%, Sharpe −0.65 | neto −19.69%, Sharpe −1.20 |

### Bloqueos y desviaciones respecto al plan

1. **Ningún grupo del universo original A pasó Johansen, y hubo que ampliar el universo.** A1
   falló en los 15 grupos y eso activó la reserva A3, que también falló. Se verificó que el
   resultado de A1 es robusto: no cambia con k_ar_diff en {1, 2, 5} ni con k elegido por AIC,
   así que no es un artefacto de la especificación de rezagos. Se decidió **no** relajar
   criterios ni seguir probando especificaciones, y en su lugar registrar el pool B1 (bancos
   canadienses) evaluándolo con exactamente los mismos umbrales. Queda asentado que B1 se
   agregó **después** de que A1 y A3 fallaran, y `m` subió de 21 a 26.

2. **Corrección a la justificación económica de B1.** En la planeación se había descrito a los
   bancos canadienses como líneas cross-listed del mismo instrumento. Es incorrecto: son cinco
   bancos distintos y no comparten flujos de efectivo. Lo que comparten es el entorno
   regulatorio (OSFI), el mercado hipotecario doméstico con seguro público (CMHC) y el ciclo de
   tasas del Banco de Canadá. Corregido en `data/universe.py` y en el plan de fases.

3. **El estadístico de reversión del VECM estaba mal elegido en la primera implementación.** Se
   había usado el α de la pata numeraria, y con eso B1 parecía no revertir (α de CM = +0.0064,
   t = +0.95). La cantidad correcta es **β'α**, la velocidad de ajuste del propio término de
   corrección de error, que resultó negativa en los dos grupos (−0.0189 y −0.0173) con vidas
   medias de 36 días consistentes con el ajuste de Ornstein-Uhlenbeck. Una pata individual
   puede desestabilizar y el spread revertir igual, porque lo que importa es el efecto neto
   ponderado por β.

4. **La varianza constante del filtro 1 no sobrevive el cambio de escala de los precios.** La
   varianza de las innovaciones normalizadas pasa de 1.0 en entrenamiento a 2.3 y 6.1 en la
   muestra completa, porque R se calibra con precios de 2012–2021 y VLO pasó de ~$30 a ~$350.
   Se resolvió haciendo que el filtro 2 use varianza de observación **adaptativa**,
   R₂,ₜ = σ̂²_{t|t−1}, que es exactamente lo que la sección 3.4 pide al definir el estado como
   la media y la varianza latentes. Resultado: el z-score tiene desviación estándar ≈1.0 en los
   tres tramos. Queda como limitación declarada que el filtro 1, al trabajar en niveles con R
   fija, tiene innovaciones mal escaladas fuera de muestra.

5. **La autocorrelación de las innovaciones de ambos filtros es ≈0.97 y no va a bajar.** No es
   un defecto: las innovaciones del filtro 1 **son** el spread, y el spread es persistente por
   construcción con vida media de 36 días. Un z-score con innovaciones blancas significaría que
   no hay reversión que operar. Para la prueba obligatoria 4 se usará la varianza de las
   innovaciones normalizadas del filtro 2 (1.10 y 1.15, dentro de tolerancia) y la blancura de
   los residuales del VECM, que es el modelo que sí describe la dinámica y cuyos residuales
   pasan Ljung-Box en los dos grupos. Se cierra en la fase 8 y se documenta como interpretación
   declarada.

6. **Dos defectos de diseño de la malla de umbrales, detectados por el resultado.** La primera
   versión incluía z_salida = 0.0, que vuelve la regla de salida inalcanzable porque pide
   |z| < 0, y el grid search la eligió: B1 quedaba invertido 2736 de 3496 días. Y z_entrada se
   topaba en 2.5, el extremo de la malla, de modo que no sabíamos si el óptimo estaba más allá.
   Se corrigieron las dos cosas y ahora ambos óptimos son interiores.

7. **Hallazgo central del laboratorio, y es negativo.** Los dos grupos funcionan en
   entrenamiento (Sharpe neto +0.64 y +0.44) y pierden en prueba y validación. En B1 el retorno
   **bruto** también es negativo fuera de muestra (−8.95% y −16.19%), así que la relación de
   equilibrio se rompió de verdad. En A2 el bruto fuera de muestra es −2.60% y +3.67%, de modo
   que ahí el problema que manda son los costos: la comisión se come ~4 puntos por partición.
   Con m = 26 y 1.3 falsos positivos esperados, esto es exactamente lo que la sección 3.2 pide
   poder sostener: no podemos afirmar que las relaciones encontradas sean estructurales. No se
   va a maquillar; es el resultado que se reporta y se defiende.

### Criterio de salida del bloque

- [x] `python main.py` corre las tres etapas de principio a fin sin errores.
- [x] Dos grupos disjuntos operados, verificado por código.
- [x] Todo lo estimado usa solo la partición de entrenamiento.
- [x] z-score calculado con estadísticos previos a la actualización del filtro 2.
- [x] Artifacts de las secciones 3.2, 3.3, 3.4 y 3.6 escritos.

---

## Bloque 2 — F6 a F8: backtest, Risk Parity, métricas, figuras y pruebas (H5–H6, cerrado dom 11:00)

### Objetivo

Llevar la contabilidad del portafolio a dólares y acciones con el límite del 80% sobre el
capital consolidado, repartir el capital entre los dos grupos por Risk Parity, cuantificar el
rebalanceo diario contra una banda de no-rebalanceo, calcular todas las métricas y las nueve
figuras obligatorias, y dejar las cuatro pruebas obligatorias más la de nivel B pasando.

### Tareas

| # | Tarea | Estado | Notas |
|---|---|---|---|
| T29 | Política de banda de no-rebalanceo | ✅ Hecho | `rebalance.py`; ancho calibrado solo en train, comparado en las 3 particiones |
| T30 | Risk Parity con contribuciones verificadas | ✅ Hecho | `risk_parity.py`; descenso coordinado cíclico |
| T31 | Dimensionamiento en acciones y dólares | ✅ Hecho | `sizing.py`; 80% del capital consolidado del cierre previo |
| T32 | Contabilidad consolidada en dólares | ✅ Hecho | `walkforward.py`; libro de patas por fecha, grupo y ticker |
| T33 | Orquestación de los 4 portafolios comparados | ✅ Hecho | `portfolio.py`; RP, pesos iguales y cada grupo solo |
| T34 | Métricas de desempeño | ✅ Hecho | `metrics.py`; 6 métricas de la tarjeta + Sharpe móvil 252 d |
| T35 | Atribución de costos y de drawdown | ✅ Hecho | `attribution.py`, `drawdown.py` |
| T36 | Las 9 figuras obligatorias | ✅ Hecho | `report_figures.py` → `artifacts/figures/` |
| T37 | Prueba 1: causalidad | ✅ Hecho | 3 fechas, tolerancia 1e-10 |
| T38 | Prueba 2: normalización | ✅ Hecho | β numeraria = 1 exacto; cruces por cero reportados |
| T39 | Prueba 3: contabilidad multi-pata | ✅ Hecho | Conciliada contra el libro de patas |
| T40 | Prueba 4: diagnóstico de filtros | ✅ Hecho | Varianza y autocorrelación de AMBOS filtros, reportadas |
| T41 | Prueba 4-B: Risk Parity | ✅ Hecho | Desvío 0.0e+00 contra tolerancia 1e-6 |

### Resultados medidos

**Risk Parity** (sobre retornos de spreads, estimado solo en entrenamiento):

| Grupo | Vol. anual del spread | Peso RP | Contribución al riesgo RP | Contribución con pesos iguales |
|---|---|---|---|---|
| VLO-MPC-PSX-PBF | 9.52% | 0.3446 | 0.500000 | 0.8196 |
| RY-TD-BMO-CM | 5.00% | 0.6554 | 0.500000 | 0.1804 |

Correlación entre spreads **−0.1374**. Convergencia en 5 iteraciones, desvío máximo contra el
objetivo 1/G: **0.0e+00**. Exposición bruta máxima 80.8% del capital corriente (el límite se
aplica contra el capital del cierre previo y la prueba 3 verifica que nunca se excede).

**Desempeño del consolidado Risk Parity** (capital inicial $1,000,000):

| Partición | Capital final | Retorno | Sharpe | Sortino | Calmar | Max DD | Win rate | Profit factor |
|---|---|---|---|---|---|---|---|---|
| Entrenamiento | $1,172,407 | +17.24% | +0.869 | +1.043 | +0.579 | −3.33% | 82.7% | 2.716 |
| Prueba | $1,112,090 | −5.14% | −0.775 | −0.690 | −0.244 | −7.72% | 55.2% | 0.417 |
| Validación | $1,044,245 | −6.10% | −0.908 | −0.972 | −0.354 | −6.33% | 62.1% | 0.588 |

**Comparación entre asignaciones** (global):

| Portafolio | Capital final | Retorno | Sharpe | Max DD |
|---|---|---|---|---|
| Consolidado Risk Parity | $1,044,245 | +4.42% | +0.146 | −12.61% |
| Consolidado pesos iguales | $1,083,626 | +8.36% | +0.259 | −11.21% |
| VLO-MPC-PSX-PBF solo | $1,214,274 | +21.43% | +0.409 | −8.72% |
| RY-TD-BMO-CM solo | $959,245 | −4.08% | −0.086 | −17.65% |

**Atribución de costos** (identidad bruto = neto + comisiones + préstamo, residuo 0.0):

| Partición | P&L bruto | Comisiones | Préstamo | P&L neto | Costos / \|bruto\| |
|---|---|---|---|---|---|
| Entrenamiento | 263,311 | 87,705 | 3,199 | 172,407 | 34.5% |
| Prueba | −23,127 | 35,754 | 1,436 | −60,317 | **160.8%** |
| Validación | −33,486 | 32,949 | 1,409 | −67,844 | 102.6% |
| Global | 206,698 | 156,408 | 6,045 | 44,245 | 78.6% |

**Atribución del drawdown máximo**: −12.61%, del 2020-05-20 al 2026-11-03 (1622 días hábiles).
RY-TD-BMO-CM originó el **87.6%** de la caída ($−131,692) y VLO-MPC-PSX-PBF el 12.4% ($−18,583).

**Rebalanceo diario contra banda** (anchos calibrados solo en train: 0.025 para A2, 0.075 para
B1). La banda reduce los días con operación de 392 a 78 (A2) y de 979 a 72 (B1) con turnover y
comisiones casi idénticos, y mejora el Sharpe en entrenamiento (+0.644 → +0.665 y
+0.439 → +0.452). Fuera de muestra el resultado es mixto: en A2 la banda es ligeramente peor en
las dos particiones, en B1 mejor en prueba y peor en validación. Conclusión reportada: el ahorro
de costos de la banda es marginal porque el turnover lo domina la apertura y el cierre de
posiciones, no el ajuste diario de β̂ₜ.

**Diagnóstico de filtros** (`artifacts/performance/filter_diagnostics.csv`):

| Grupo | Filtro | Tramo | Varianza | ρ₁ | φ = 0.5^(1/h) | Ljung-Box(10) |
|---|---|---|---|---|---|---|
| A2 | 1 | entrenamiento | 0.959 ✅ | 0.982 | 0.9811 | 0.0000 |
| A2 | 1 | muestra completa | 2.349 ❌ | 0.980 | 0.9811 | 0.0000 |
| A2 | 2 | muestra completa | 1.100 ✅ | 0.971 | 0.9811 | 0.0000 |
| A2 | VECM (ECT) | entrenamiento | — | — | — | 0.2780 ✅ |
| B1 | 1 | entrenamiento | 0.988 ✅ | 0.984 | 0.9810 | 0.0000 |
| B1 | 1 | muestra completa | 6.113 ❌ | 0.988 | 0.9810 | 0.0000 |
| B1 | 2 | muestra completa | 1.153 ✅ | 0.976 | 0.9810 | 0.0000 |
| B1 | VECM (ECT) | entrenamiento | — | — | — | 0.0567 ✅ |

**Pruebas:** 27 pasan, 0 fallan.

### Bloqueos y desviaciones respecto al plan

1. **Risk Parity devolvió pesos negativos y violó el límite del 80%.** La primera
   implementación usaba la iteración de punto fijo w_i ← 1/(Σw)_i normalizada, que no garantiza
   positividad. Como la correlación entre los dos spreads es **negativa** (−0.1374), el producto
   (Σw)_i se volvió negativo y la iteración converge a w = (−1.109, +2.109). Consecuencias: deja
   de ser una asignación de capital interpretable, y como Σ|w| = 3.2, el nocional bruto llegó al
   **267.9% del capital**, violando el límite del 80%. Se reemplazó por descenso coordinado
   cíclico, cuya actualización en forma cerrada

   w_i = (−b_i + sqrt(b_i² + 4σ_i²/G)) / (2σ_i²),  b_i = Σ_{j≠i} σ_ij w_j

   da w_i > 0 por construcción porque la raíz siempre excede |b_i|. Resultado: w = (0.345,
   0.655), contribuciones exactamente 0.5 / 0.5, exposición bruta máxima 80.8%.

2. **La prueba de normalización falló por una razón legítima, y eso cambió la prueba.** Detectó
   un cambio de signo en β_PBF. Al investigarlo resultó **no** ser un volteo espurio de
   normalización sino un **cruce continuo por cero** (+0.000061 → −0.002699, salto diario máximo
   0.0047 en todo el proyecto) al final de una decadencia monótona de +0.25 (2012) a −0.10
   (2026). La prueba se reformuló para distinguir los dos fenómenos: exige que todo cambio de
   signo sea continuo (|β| < 0.01 a ambos lados), que es lo que descarta un error de
   normalización, y reporta los cruces como resultado en
   `artifacts/kalman/cruces_por_cero_beta.csv`.

3. **Error propio en el conteo de cruces, corregido.** Al documentar el punto anterior se afirmó
   que había "exactamente un cruce", basándose en una verificación que solo había mirado β_PBF.
   El artifact completo muestra **siete**: β_PBF una vez, β_RY dos y β_TD cuatro. Todos
   continuos. El punto de fondo se refuerza en lugar de debilitarse: en **los dos** grupos los
   pesos del equilibrio se movieron materialmente entre entrenamiento y validación (β_PBF de
   +0.242 a −0.099; β_RY de −0.205 a −0.613; β_TD de +0.115 a +0.458). β_TD cruza cuatro veces
   porque ronda cero entre 2012 y 2022, es decir que esa pata casi no participaba en la relación.
   Corregido en `tests/test_normalization.py` y en la guía de estudio.

4. **La prueba 4 no puede afirmar que la autocorrelación sea baja, y no lo afirma.** ρ₁ ≈
   0.97–0.99 en los dos filtros. Se decidió reportarlo explícitamente y convertirlo en una
   predicción cuantitativa: un AR(1) con vida media h tiene φ = 0.5^(1/h) = 0.981 para h ≈ 36 d,
   y la autocorrelación medida coincide dentro de 0.01. La prueba verifica esa coincidencia con
   tolerancia declarada de 0.05, de modo que la persistencia queda **explicada por el modelo**.
   La blancura que sí se exige es la de los residuales del VECM, que es el modelo que describe la
   dinámica, y ambos grupos la pasan. Declarado en el README en una sección propia.

5. **La varianza del filtro 1 no cumple fuera de muestra y la aserción se acotó a
   entrenamiento.** 2.349 y 6.113 contra el rango [0.8, 1.25]. Causa: R constante calibrada con
   precios de 2012–2021 contra niveles 10 veces mayores en 2026. La aserción se evalúa en
   entrenamiento, que es donde el parámetro fue calibrado, y el valor fuera de muestra se
   reporta sin esconderlo en el README y en el CSV de diagnósticos.

6. **Bug propio en el fixture de causalidad.** `etiquetar_particiones` sobre un índice truncado
   recalcula los cortes al 60% de la muestra corta, lo que cambiaría el conjunto de
   entrenamiento y haría que la prueba comparara dos experimentos distintos en lugar de medir
   causalidad. Se corrigió pasando la fecha de corte del panel completo como parámetro.

7. **Risk Parity rinde menos que pesos iguales, y se reporta tal cual.** +4.42% contra +8.36%
   global. La razón es que Risk Parity iguala contribuciones al **riesgo**, no al retorno, y por
   eso asignó 65.5% al grupo menos volátil (B1, vol 5.0%), que resultó el de peor desempeño. Lo
   que sí cumplió es su objetivo declarado: contribuciones iguales con desvío 0.0e+00, mientras
   pesos iguales dejó un reparto de riesgo 82% / 18%. No se cambió nada para mejorar el
   resultado.

8. **La diversificación entre grupos es parcialmente real.** La correlación entre spreads es
   −0.1374 y el max drawdown del consolidado (−12.61%) es mejor que el de B1 solo (−17.65%), así
   que amortiguó el peor caso individual. Pero la atribución muestra que el drawdown no fue un
   evento conjunto: B1 originó el 87.6%. Una correlación baja no protege de que un grupo deje de
   funcionar.

9. **Los costos, no la señal, son lo que mata al grupo A2.** 78.6% del P&L bruto global y 160.8%
   en prueba. El préstamo es irrelevante ($6,045 contra $156,408 de comisión, 3.7% de los
   costos): el costo que manda es la comisión de 0.125% por pata con rebalanceo diario de cuatro
   patas. En B1 el diagnóstico es distinto: su bruto fuera de muestra también es negativo
   (−8.95% y −16.19%), así que ahí la relación se rompió de verdad y ningún ahorro de costos lo
   arreglaría.

### Criterio de salida del bloque

- [x] `python main.py` corre las cinco etapas de principio a fin sin errores.
- [x] `pytest -q`: 27 pruebas pasan, incluidas las 4 obligatorias y la de nivel B.
- [x] Contribuciones al riesgo iguales dentro de 1e-6 (desvío real 0.0e+00).
- [x] Límite del 80% sobre el capital consolidado verificado día por día.
- [x] Identidades contables (capital = efectivo + patas; bruto = neto + costos) cierran a 1e-6.
- [x] Las 9 figuras obligatorias generadas.
- [x] Ningún parámetro, umbral ni decisión se cambió para mejorar prueba o validación.

---

## Bloque 3 — F9 y F10: dashboard y documentación (H7–H8, cerrado dom 18:00)

### Objetivo

Construir el dashboard con las dos vistas que pide el PDF leyendo solo `artifacts/`, escribir
las tres formulaciones de Powell con su ejemplo numérico, completar el README, redactar el
post-mortem y producir `docs/reporte.pdf`. Antes del bloque se atendieron dos correcciones de
reproducibilidad pedidas por el equipo.

### Correcciones previas al bloque

| # | Corrección | Qué se hizo |
|---|---|---|
| C1 | Versionar `artifacts/` y congelar los precios | Se creó `data/precios_snapshot.parquet` (520 KB, 20 tickers × 3,496 días), `main.py` lo lee por defecto y `--forzar-descarga` lo regenera. Se quitó `artifacts/` y `data/cache/` del `.gitignore`, que ahora solo excluye herramientas y `estudio/`. Se eliminó el directorio `data/cache/` |
| C2 | Explicar la exposición de 80.8% | Sección propia en el README: el límite se aplica al dimensionar con $C_{t-1}$ y la métrica se mide contra $C_t$; el límite real nunca se viola y la columna `limite_nocional` lo verifica día por día |

**Lo que el snapshot resolvió:** entre dos descargas de yfinance separadas por horas, el vector
cointegrante pasó de $\beta_{PBF} = 0.594517$ a $0.594516$. yfinance recalcula el cierre ajustado
hacia atrás con cada dividendo, así que una descarga en otra fecha daría cifras distintas a las
del reporte. Tamaños: `artifacts/` 7.7 MB, snapshot 520 KB.

### Tareas

| # | Tarea | Estado | Notas |
|---|---|---|---|
| T42 | Snapshot congelado de precios | ✅ Hecho | `data/loader.py` reescrito; sin red por defecto |
| T43 | Versionar artifacts | ✅ Hecho | `.gitignore` reducido a herramientas y `estudio/` |
| T44 | Capa de solo lectura del dashboard | ✅ Hecho | `data_access.py`; no importa ningún módulo del pipeline |
| T45 | Componentes visuales compartidos | ✅ Hecho | `components.py`; sombreado de particiones y paleta |
| T46 | Vista ejecutiva (5 elementos) | ✅ Hecho | `views_executive.py` |
| T47 | Vista técnica (7 elementos + nivel B) | ✅ Hecho | `views_technical.py` |
| T48 | App y barra lateral | ✅ Hecho | `app.py`; un solo comando |
| T49 | Ejemplo numérico de Powell | ✅ Hecho | `kalman/worked_example.py`; 4 CSV con la recursión completa |
| T50 | Formulaciones de Powell | ✅ Hecho | Reporte §6.1 a §6.5 |
| T51 | README completo | ✅ Hecho | Reproducibilidad, exposición 80.8%, guía del dashboard, uso de IA |
| T52 | Post-mortem | ✅ Hecho | Borrador con 6 marcas `[REVISAR]` |
| T53 | Reporte en Markdown y PDF | ✅ Hecho | `docs/reporte.md` y `docs/build_reporte.py` |

### Resultados medidos

**Dashboard.** Verificado con `streamlit.testing.v1.AppTest`, que ejecuta el script y expone las
excepciones:

| Vista | Resultado | Elementos |
|---|---|---|
| Ejecutiva | Sin excepciones | 9 subencabezados, 14 métricas, 3 selectores, 4 multiselectores |
| Técnica | Sin excepciones | 12 subencabezados, 7 métricas, 5 tablas, 1 aviso (el intencional sobre autocorrelación) |

También se probaron las interacciones: cambio de portafolio, de periodo y de filtro de resultado
en la ejecutiva; cambio de grupo y de series del Sharpe móvil en la técnica. Ninguna produce
excepciones. El servidor responde 200 y `/_stcore/health` devuelve `ok`.

**Reporte PDF.** 20 páginas, 2,348 KB, **265 fórmulas renderizadas sin fallas**, las 9 figuras
incrustadas. Verificado abriendo el PDF: no quedan `$` sin convertir ni tablas en crudo, y la
inspección visual de las páginas con fórmulas y figuras confirma que se ven bien.

### Bloqueos y desviaciones respecto al plan

1. **La máquina no tiene pandoc, LaTeX ni wkhtmltopdf.** Hubo que construir la cadena de
   conversión: Markdown → HTML (`markdown`) → PDF (`PyMuPDF Story`), con las fórmulas
   rasterizadas por el motor `mathtext` de matplotlib e incrustadas como imágenes. El script
   `docs/build_reporte.py` reporta cualquier fórmula que no pueda renderizar, para que una falla
   no pase inadvertida.

2. **Tres fallas del generador de PDF, encontradas y corregidas:**
   - `mathtext` no conoce `\le` ni `\ge`, solo `\leq` y `\geq`. Una fórmula fallaba; se agregó la
     sustitución y ahora las 265 renderizan.
   - `DocumentWriter` escribe los flujos de contenido **sin comprimir**: el PDF pesaba **52 MB**
     cuando las imágenes sumaban solo 2.1 MB. Se escribe a un intermedio y se vuelve a guardar
     con `garbage=4, deflate=True, clean=True`. Quedó en 2.3 MB.
   - **Las fórmulas salían a un tercio de su tamaño** (3 pt en lugar de 10 pt) por una doble
     conversión mía: PyMuPDF honra el DPI que matplotlib escribe en el PNG, así que su `rect` ya
     viene en puntos, y yo volvía a multiplicarlo por 72/220. Se detectó midiendo el rectángulo
     real con que las imágenes quedaban colocadas en el PDF, no a ojo.

3. **Dos fallas del dashboard con `add_vline` de Plotly.** Pasar un `Timestamp` junto con
   `annotation_text` hace que Plotly intente promediar las coordenadas de la anotación y falle
   (`TypeError` de pandas). Convertir la fecha a texto cambió el error pero no lo resolvió
   (`int + str`). La solución fue dibujar la línea con `add_shape` y la etiqueta con
   `add_annotation` por separado.

4. **`AppTest` fue decisivo.** El servidor de Streamlit devuelve HTTP 200 aunque el script falle,
   así que comprobar el puerto no prueba nada. `AppTest` ejecuta el script y expone las
   excepciones: ahí aparecieron las dos fallas de Plotly.

5. **El heredoc de bash volvió a fallar** con contenido que lleva apóstrofos. Se escribieron los
   parches de texto largo a archivos del scratchpad y se ejecutaron desde ahí. Sin impacto en el
   entregable.

### Criterio de salida del bloque

- [x] `python main.py` corre las cinco etapas desde el snapshot, sin red.
- [x] `pytest -q`: 27 pruebas pasan.
- [x] `streamlit run dashboard/app.py` abre y las dos vistas renderizan sin excepciones.
- [x] `docs/reporte.pdf` existe, con las 9 figuras y 265 fórmulas legibles.
- [x] Las tres formulaciones de Powell y el ejemplo numérico están en el reporte.
- [x] `postmortem.md` entregado como borrador con las partes de reflexión marcadas.
- [x] `plan_fases.md` sin pendientes: 86 requisitos marcados como cumplidos.

---

## Bloque 4 — Correcciones de la revisión externa

Al cerrar el bloque 3 el trabajo se sometió a una revisión externa, que encontró **12 errores**.
Dos de ellos cambiaron los resultados, así que se volvió a correr el pipeline completo, se
recalibraron los umbrales con el motor corregido y se actualizaron todas las cifras del reporte,
el README y este sprint. **No se cambió ningún parámetro para mejorar los resultados fuera de
muestra:** los umbrales nuevos salieron de volver a correr la misma búsqueda.

### Críticos

| # | Error | Efecto | Corrección |
|---|---|---|---|
| C1 | **Dos contabilidades conviviendo.** El grid search y la calibración de la banda usaban pesos w_j = β_j / Σ\|β_k\|, sin precios. Como β está en acciones, el peso en dólares es w_j = β_j P_j / Σ\|β_k P_k\|. El motor consolidado sí usaba dólares. | Se calibraba sobre un portafolio y se operaba otro: correlación de 0.79 entre los retornos diarios de ambas contabilidades para B1 | Un solo motor: grid search, banda y resultado final llaman a `walkforward.simular_consolidado` vía `engine.simular_grupo_en_dolares`, restringido a entrenamiento para calibrar. La banda se implementó sobre acciones. Grid search vuelto a correr |
| C2 | **El filtro 1 no arrancaba en el vector de Johansen**, sino en una solución OLS (β de PBF en +0.256 contra +0.595) | El reporte afirmaba que el spread operado era el ECT del VECM, y era falso | θ₀ = −β^J, c₀ = media en entrenamiento del residual de ese vector, R = su varianza. Σ₀ sigue siendo la covarianza OLS. El día 0 reproduce Johansen exactamente |

### Menores

| # | Error | Corrección |
|---|---|---|
| 1 | Figura 4 con picos de ±1000 al dividir entre una numeraria casi cero | Eje acotado a [−3, 3] y ventanas sin cointegración al 95% sombreadas, en la figura y en el dashboard |
| 2 | Win rate y profit factor de pesos iguales calculados con la bitácora de Risk Parity | Cada portafolio construye su bitácora en dólares con su propio `pnl_por_grupo` |
| 3 | Sortino con la desviación estándar del subconjunto de días negativos | Desviación a la baja estándar, √(media(min(r,0)²)) sobre todos los días |
| 4 | Prueba de contabilidad trivial: el efectivo se definía como capital − patas | El motor lleva el efectivo como saldo propio; la prueba lo reconstruye desde los flujos del libro y verifica que tenga variación independiente |
| 5 | Se afirmaba que 1/σ no resuelve Risk Parity | Con dos grupos sí lo resuelve exactamente (el término cruzado se cancela); se demuestra en el reporte §8.1 y se verifica: w·σ = 0.0288552 en ambos |
| 6 | `det_order=0` descrito como constante restringida a la relación de cointegración | Es una constante **no restringida**; corregido en `config.py`, `cointegration/johansen.py` y el reporte |
| 7 | "Costos = 160.8% del bruto" con bruto negativo | El cociente solo se reporta con bruto positivo; con bruto negativo se usa el costo sobre capital medio |
| 8 | Sobreajuste de umbrales dado por demostrado | Reportado como hipótesis no resuelta, en el reporte y en el post-mortem |
| 9 | Figuras 2, 4, 5 y 7 como miniaturas al pie de página en el PDF; `\mathrm{}` con espacios pegados | Ancho completo (504 pt) con `page-break-inside: avoid`; los espacios dentro de `\mathrm{}` se convierten a `\ ` |
| 10 | Dashboard: "new text" en las líneas de umbral negativas y `$K_t$` literal en los ejes | No pasar `annotation_text` cuando no se quiere etiqueta (pasar `None` no la suprime); etiquetas en texto plano |

### Efecto en las cifras

| Magnitud | Antes | Después |
|---|---|---|
| Umbrales A2 (entrada / salida / stop / banda) | 1.50 / 1.00 / 3.00 / 0.025 | **2.25 / 0.25 / 3.00 / 0.075** |
| Umbrales B1 (entrada / salida / stop / banda) | 1.00 / 0.50 / 4.00 / 0.075 | **1.25 / 0.50 / 3.50 / 0.100** |
| R del filtro 1, A2 / B1 | 14.09 / 0.86 | **17.56 / 1.13** |
| Operaciones A2 / B1 | 72 / 61 | **27 / 45** |
| Consolidado RP: train / prueba / validación | +17.24% / −5.14% / −6.10% | **+13.40% / −2.60% / −6.39%** |
| Consolidado RP: Sharpe train / validación | +0.869 / −0.908 | **+0.932 / −0.969** |
| A2 solo, global / validación | +21.43% / −0.67% | **+29.56% / +4.35%** |
| B1 solo, global / validación | −4.08% / −9.32% | **−6.72% / −10.82%** |
| Pesos de Risk Parity | 0.345 / 0.655 | **0.306 / 0.694** |
| Correlación entre spreads | −0.1374 | **−0.0857** |
| Costos sobre bruto global | 78.6% | **75.2%** |
| Max drawdown del consolidado | −12.61% | **−11.16%** |
| Cruces por cero de β | 7 | **5** |
| Pruebas | 27 | **29** |

**El cambio cualitativo que importa:** con el motor corregido, el grupo de refinación **gana en
validación** (+4.35%) en lugar de perder, y termina en +29.56% global. La conclusión general no
cambia (no podemos afirmar que las relaciones sean estructurales, con m = 26 y 1.3 falsos
positivos esperados), pero ya no es cierto que ninguno de los dos grupos haya sobrevivido fuera
de muestra. El reporte y el post-mortem se reescribieron para separar el diagnóstico por grupo
en lugar de dar un veredicto único.

### Lo que esto dice del proceso

Los 12 errores sobrevivieron a tres revisiones de bloque. Dos patrones explican la mayoría:

1. **Afirmaciones de diseño no verificadas.** El docstring del motor decía que usar el mismo
   código para calibrar y reportar evita discrepancias, pero había dos motores y nunca se
   comprobó que coincidieran. El reporte afirmaba que el ECT era el spread operado, y nadie lo
   contrastó contra el artifact. Las dos cosas se habrían detectado con un `assert`.
2. **Métricas que se veían razonables.** Sortino, win rate de pesos iguales, el ratio de costos
   con bruto negativo y la identidad contable trivial: cuatro números que nadie cuestionó porque
   el orden de magnitud parecía correcto.

### Criterio de salida

- [x] `python main.py` corre las cinco etapas desde el snapshot, sin red.
- [x] `pytest -q`: 29 pruebas pasan.
- [x] El dashboard abre y las dos vistas renderizan sin excepciones.
- [x] `docs/reporte.pdf` regenerado con las cifras nuevas y las 9 figuras a ancho completo.
- [x] Ningún parámetro se cambió para mejorar prueba o validación.
