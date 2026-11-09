# Lab 03 MyST — Arbitraje Estadístico con Cointegración y Filtros de Kalman

**ITESO — Microestructuras y Sistemas de Trading · Módulo 3 · Equipo 1**

## Integrantes

| Integrante | Rol principal |
|---|---|
| Rogelio Adrián Arroyo Valencia | Datos, cointegración (Engle-Granger / Johansen), VECM y lógica de señales |
| Ernesto Andrés González Lomelí | Filtros de Kalman, motor de backtest, Risk Parity, dashboard y pruebas |

El desglose completo de responsabilidades está en [`project_management/charter.md`](project_management/charter.md).

## Nivel de alcance

**Nivel B.** Incluye todo el alcance base (selección de un grupo cointegrado, dos filtros de
Kalman, VECM, backtest con costos y dashboard) **más** la operación simultánea de varios
grupos cointegrados disjuntos con asignación de capital por **Risk Parity** (sección 3.7 de
los lineamientos).

## Descripción del proyecto

Estrategia de arbitraje estadístico sobre portafolios multi-activo de reversión a la media.
En lugar de operar un par, se identifican grupos de **4 activos** (= 2 · 2 integrantes) que
comparten una tendencia estocástica común, se extrae el vector del portafolio reversivo con
la prueba de **Johansen**, se estiman pesos dinámicos con un **filtro de Kalman**, y se
generan señales a partir de las desviaciones del equilibrio normalizadas por un **segundo
filtro de Kalman** que estima la media y la varianza latentes del spread en tiempo real.
Ambos filtros se formulan explícitamente como procesos de decisión secuencial bajo el marco
de Powell.

### Universo evaluado y grupos operados

El universo se **pre-registra** en [`data/universe.py`](data/universe.py) antes de calcular
cualquier estadístico, y el campo `momento_registro` de cada pool deja asentado cuándo entró y
por qué. Es la defensa del equipo frente al problema de pruebas múltiples.

| Pool | Nombre | Tickers | Tendencia estocástica común | Cuándo se registró |
|---|---|---|---|---|
| A1 | E&P / integradas | XOM, CVX, COP, OXY, EOG, DVN | Nivel del crudo (WTI / Brent) | Original |
| A2 | Refinación | VLO, MPC, PSX, PBF, DINO | Margen de refinación (crack 3-2-1) | Original |
| A3 | Servicios petroleros | SLB, HAL, NOV, RIG | Ciclo global de capex upstream | Reserva, activada al fallar A1 |
| B1 | Bancos canadienses en NYSE | RY, TD, BNS, BMO, CM | Ciclo del Banco de Canadá y mercado hipotecario doméstico | **Después** de que A1 y A3 fallaran |

**Resultado del embudo: de 26 grupos evaluados, solo 2 pasaron.** Ningún grupo de A1 pasó
Johansen al 95% (traza máxima 32.06 contra crítico 47.85), lo que activó la reserva A3, que
tampoco pasó (EG p = 0.624). Verificamos que el fracaso de A1 es robusto a la especificación de
rezagos ($k \in \{1, 2, 5\}$ y $k$ por AIC) y **no relajamos ningún criterio**: en su lugar
registramos B1 y lo evaluamos con exactamente los mismos umbrales.

| Grupo operado | Pool | Pata numeraria | β | ADF del spread | Vida media |
|---|---|---|---|---|---|
| VLO-MPC-PSX-PBF | A2 | VLO | (1, −0.787, −0.884, +0.595) | p = 0.0004 | 37.3 d |
| RY-TD-BMO-CM | B1 | CM | (−0.338, +0.562, −0.559, 1) | p = 0.0056 | 33.1 d |

Son disjuntos en activos, verificado por código (`verificar_pools_disjuntos`).

**Sobre B1:** son **cinco bancos distintos**, no el mismo instrumento en dos bolsas. Cada uno
está cross-listed individualmente en NYSE y TSX, pero entre ellos no comparten flujos de
efectivo. Lo que comparten es el entorno: el mismo regulador prudencial (OSFI), el mismo mercado
hipotecario doméstico con seguro público (CMHC) y el mismo ciclo de tasas del Banco de Canadá.
Ese mecanismo es nombrable y falsable.

**Defensa frente a pruebas múltiples, con cifras propias.** Con $\alpha = 0.05$ y $m$ hipótesis,
el número esperado de falsos positivos bajo la nula es $\mathbb{E}[\text{FP}] = \alpha\,m$:

| Escenario | $m$ | $\mathbb{E}[\text{FP}]$ |
|---|---|---|
| Nuestro universo pre-registrado | 26 | **1.3** |
| Grupos de 4 en el S&P 500, sin restricción | $\binom{500}{4} \approx 2.57\times10^{9}$ | $\approx 1.3\times10^{8}$ |

Pasaron 2 grupos y esperábamos 1.3 espurios. **No podemos descartar que uno de los dos sea un
hallazgo espurio**, y eso es exactamente lo que el desempeño fuera de muestra sugiere. No
aplicamos correcciones de multiplicidad tipo Bonferroni porque no se vieron en clase; reportamos
$m$ y $\alpha m$, que es lo que piden los lineamientos.

### Parámetros fijos

| Parámetro | Valor | Nota |
|---|---|---|
| Ventana histórica | 2012-12-13 → 2026-11-06 (**13.90 años**) | La impone PBF, el ticker más joven del universo |
| División | 60 / 20 / 20 cronológica | 2097 / 699 / 700 días |
| Activos por grupo | 4 | 2 · n con n = 2 integrantes |
| Comisión | 0.125% | Por operación, por pata, a la entrada y a la salida |
| Tasa de préstamo | 0.25% anual | **actual/365** sobre días calendario, nocional en corto |
| Anualización de métricas | **252** días hábiles | Deliberadamente distinta del devengo |
| Capital inicial | $1,000,000 | |
| Exposición máxima | 80% del capital **consolidado** | Medida como nocional bruto |
| Semilla aleatoria | **42** | `config.SEMILLA`, fijada en `config.fijar_semilla()` |

Todas las constantes viven en un solo lugar: [`config.py`](config.py).

## Instalación

Requiere Python 3.12 (entorno de desarrollo: 3.12.4).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Reproducción

Comando exacto, un solo paso:

```bash
python main.py
```

El pipeline lee los precios del **snapshot congelado** `data/precios_snapshot.parquet`, limpia
y valida el panel, aplica la partición cronológica y escribe todas sus salidas en `artifacts/`.
No necesita conexión a internet.

Pruebas:

```bash
pytest -q
```

Dashboard:

```bash
streamlit run dashboard/app.py
```

Regenerar el reporte en PDF a partir de `docs/reporte.md`:

```bash
python docs/build_reporte.py
```

Esta máquina no tiene pandoc ni LaTeX, así que el PDF se arma con la cadena
Markdown → HTML (`markdown`) → PDF (`PyMuPDF Story`), con las fórmulas rasterizadas por el
motor `mathtext` de matplotlib. El script reporta cualquier fórmula que no haya podido
renderizar, de modo que una falla no pase inadvertida.

## Reproducibilidad: por qué versionamos el snapshot y los artifacts

### El snapshot de precios

**yfinance no devuelve la misma serie histórica en dos fechas distintas.** El cierre ajustado
se recalcula hacia atrás cada vez que una empresa paga un dividendo o hace un split: el factor
de ajuste de *toda* la historia previa cambia. Lo comprobamos en este propio proyecto: entre dos
descargas separadas por horas, el vector cointegrante pasó de
$\beta_{PBF} = 0.594517$ a $0.594516$. Es un cambio diminuto, pero sobre 3,496 observaciones y
dos filtros recursivos se propaga, y una descarga hecha un mes después daría umbrales y cifras
distintas a las que este reporte afirma.

Por eso la descarga se congeló **una sola vez** en `data/precios_snapshot.parquet` (520 KB, 20
tickers × 3,496 días), ese archivo **se versiona** en el repositorio, y `python main.py` lo usa
por defecto. Quien clone el repositorio obtiene exactamente las cifras del reporte, sin red y
sin depender de que yfinance siga respondiendo.

Para regenerarlo desde yfinance:

```bash
python main.py --forzar-descarga
```

Es la única forma de actualizar el snapshot, y hacerlo **cambia los resultados**. Si alguien lo
corre, las cifras del reporte dejan de coincidir con las de `artifacts/`.

### Los artifacts

`artifacts/` (7.7 MB) **también se versiona**, en contra de la práctica habitual de ignorar
salidas generadas. La razón es el requisito de la sección 3.8: el dashboard debe cargar desde
los archivos de salida del backtest y **no volver a ejecutarlo**. Si `artifacts/` no estuviera
en el repositorio, `streamlit run dashboard/app.py` fallaría en una copia recién clonada hasta
correr el pipeline completo. Versionándolo, el dashboard abre de inmediato.

Desglose: `backtest/` 2.5 MB (libros de patas, con una fila por fecha × grupo × ticker),
`kalman/` 1.9 MB, `figures/` 1.8 MB (las 9 figuras en PNG), `prices.parquet` 520 KB, el resto
< 700 KB.

`.gitignore` solo excluye artefactos de herramientas (`__pycache__/`, `.pytest_cache/`,
entornos virtuales, archivos de sistema) y la carpeta `estudio/`, que es material interno de
preparación y no parte del entregable.

## Estructura del repositorio

```
├── config.py              # todos los parámetros y la semilla (módulo transversal)
├── artifacts_io.py        # única capa de lectura/escritura de artifacts/ (transversal)
├── main.py                # pipeline completo, un solo comando
├── data/                  # ingesta, limpieza, validación, partición 60/20/20
├── cointegration/         # screening, Engle-Granger, Johansen, spread
├── kalman/                # filtro 1 (pesos) y filtro 2 (señal)
├── strategy/              # VECM, z-score, señales, umbrales, sizing
├── backtest/              # motor con costos, walk-forward, Risk Parity
├── performance/           # métricas, atribución, figuras del reporte
├── dashboard/app.py       # dashboard (solo lee artifacts/)
├── project_management/    # charter, sprints, post-mortem
├── artifacts/             # salidas del backtest que lee el dashboard
├── tests/                 # pruebas con pytest
└── docs/                  # reporte.md, reporte.pdf, build_reporte.py
```

### Dos módulos en la raíz que los lineamientos no listan

`config.py` y `artifacts_io.py` son **transversales**: los consumen todas las carpetas del
pipeline y, en el caso de `artifacts_io.py`, también el dashboard. Si vivieran dentro de un
paquete concreto se crearían dependencias invertidas —por ejemplo `data/` tendría que
importar de `backtest/` para poder escribir en `artifacts/`—, lo que rompería el orden del
pipeline. Dejarlos en la raíz mantiene el grafo de dependencias en una sola dirección:

```
config.py  ←  data/ → cointegration/ → kalman/ → strategy/ → backtest/ → performance/
                                  ↓ (todos escriben)
                            artifacts_io.py → artifacts/ → dashboard/ (solo lee)
```

Ventaja concreta para la rúbrica: cada parámetro del laboratorio (comisión, tasa de préstamo,
semilla, umbrales, burn-in) existe en **un solo lugar** y es rastreable, y el formato y la
ruta de cada artifact están definidos en **un solo lugar**, así que el dashboard no puede
desincronizarse del backtest.

## Por qué la exposición bruta máxima medida es 80.8% y no 80.0%

El límite del 80% se aplica **al dimensionar la posición**, y el dimensionamiento usa el capital
del **cierre previo**:

$$N_{g,t} = 0.80 \cdot C_{t-1} \cdot w_g, \qquad u_{g,t} = \frac{N_{g,t}}{\sum_k |\hat\beta_{k,t}|\,P_{k,t}}, \qquad q_{j,t} = d_{g,t}\,u_{g,t}\,\hat\beta_{j,t}$$

Se usa $C_{t-1}$ y no $C_t$ para evitar una circularidad: $C_t$ todavía depende del resultado
del día, que a su vez depende del tamaño de la posición que estamos calculando.

La métrica que reportamos como "exposición" es, en cambio, el nocional bruto contra el capital
**corriente**:

$$\text{exposición}_t = \frac{\sum_g \sum_j |q_{j,t}|\,P_{j,t}}{C_t}$$

Las dos cosas difieren porque entre que se dimensiona y se mide, el capital se movió: si el día
fue malo y $C_t < C_{t-1}$, el mismo nocional representa una fracción mayor del capital nuevo.
Con un día de −1% en el capital, un 80.0% al dimensionar aparece como ~80.8% al medir.

**El límite nunca se viola.** Lo que la restricción exige es
$\sum_g\sum_j |q_{j,t}| P_{j,t} \le 0.80\,C_{t-1}$, y eso se cumple por construcción y se
verifica día por día: la serie guarda la columna `limite_nocional` $= 0.80\,C_{t-1}$ y la prueba
`tests/test_accounting.py::test_exposicion_bruta_respeta_el_limite_del_80_por_ciento` comprueba
que `nocional_bruto` nunca la excede, con tolerancia de $10^{-6}$ USD.

La alternativa sería redimensionar con $C_t$, pero eso requiere resolver un punto fijo cada día
o usar información del cierre para decidir la posición de ese mismo cierre. Preferimos la
convención causal y declarar la diferencia.

## Diagnóstico de los filtros: qué se cumple y qué no

La prueba obligatoria 4 (sección 3.11.4) exige que las innovaciones normalizadas tengan
**varianza cercana a uno** y **autocorrelación no significativa**. Reportamos las dos cosas
sin maquillarlas. Cifras producidas por `pytest` y guardadas en
`artifacts/performance/filter_diagnostics.csv`:

| Grupo | Filtro | Tramo | Varianza | ρ₁ | Ljung-Box(10) p |
|---|---|---|---|---|---|
| VLO-MPC-PSX-PBF | 1 | entrenamiento | 0.840 ✅ | 0.980 | 0.0000 ❌ |
| VLO-MPC-PSX-PBF | 1 | muestra completa | 2.140 ❌ | 0.981 | 0.0000 ❌ |
| VLO-MPC-PSX-PBF | 2 | muestra completa | 1.102 ✅ | 0.974 | 0.0000 ❌ |
| VLO-MPC-PSX-PBF | VECM (ECT) | entrenamiento | — | — | 0.2780 ✅ |
| RY-TD-BMO-CM | 1 | entrenamiento | 0.869 ✅ | 0.982 | 0.0000 ❌ |
| RY-TD-BMO-CM | 1 | muestra completa | 5.223 ❌ | 0.988 | 0.0000 ❌ |
| RY-TD-BMO-CM | 2 | muestra completa | 1.147 ✅ | 0.975 | 0.0000 ❌ |
| RY-TD-BMO-CM | VECM (ECT) | entrenamiento | — | — | 0.0567 ✅ |

### La autocorrelación NO es baja, y no puede serlo

**ρ₁ ≈ 0.97–0.99 en los dos filtros.** No lo presentamos como si se cumpliera, porque no se
cumple. La razón es estructural, no un error de implementación: las innovaciones del filtro 1
**son** el spread, y el spread es un proceso de reversión a la media con vida media de ~36
días, es decir persistente por construcción. Si estas innovaciones fueran ruido blanco, el
spread no tendría reversión que explotar y la estrategia no tendría razón de existir. El
filtro 2 resta una media latente que se mueve ~3 veces más despacio que la reversión, así que
su innovación sigue siendo la desviación del equilibrio, que es precisamente la señal.

En lugar de declarar el requisito cumplido, lo convertimos en una **predicción cuantitativa
verificable**. Un AR(1) con vida media $h$ tiene coeficiente

$$\phi = e^{-\ln 2 / h} = 0.5^{1/h}$$

Con $h = 36.3$ y $36.1$ días, $\phi = 0.9811$ y $0.9810$. La autocorrelación medida es
0.981 y 0.988 (filtro 1) y 0.974 y 0.975 (filtro 2): coincide con la predicción dentro de
0.008. `tests/test_filter.py::test_autocorrelacion_de_ambos_filtros_coincide_con_la_vida_media`
verifica esa coincidencia con tolerancia declarada de 0.05. La persistencia queda **explicada
por el modelo**, no atribuida a un defecto.

El diagnóstico de blancura que sí corresponde exigir es el de los **residuales del VECM**, que
es el modelo que describe la *dinámica* del spread (incluye el término de corrección de error
y los rezagos en diferencias). Esos sí deben ser ruido blanco y lo son: Ljung-Box p = 0.278 y
0.057, ambos por encima de 0.05.

### La varianza del filtro 1 se degrada fuera de muestra

El filtro 1 cumple varianza ≈ 1 en entrenamiento (0.840 y 0.869) pero no en la muestra
completa (2.140 y 5.223). Causa: su varianza de observación $R$ es **constante** y se calibra
con precios de 2012–2021, mientras los niveles del universo se multiplicaron por casi diez
hacia 2026 (VLO pasó de ~\$30 a ~\$350). Es una limitación conocida de trabajar en niveles
con $R$ fija, y es exactamente la razón por la que el filtro 2 usa varianza **adaptativa**
$R_{2,t} = \hat\sigma^2_{t|t-1}$, con la que el z-score mantiene desviación estándar ≈ 1.0 en
los tres tramos.

## Guía de uso del dashboard

```bash
streamlit run dashboard/app.py
```

Abre en `http://localhost:8501`. **No ejecuta el backtest**: lee únicamente los archivos de
`artifacts/`, que están versionados en el repositorio, así que funciona en una copia recién
clonada sin correr nada antes.

### Barra lateral

Es el contexto de la corrida y está siempre visible: el selector de **vista**, el ID y la fecha
de la corrida, la semilla (42), la versión de Python, la ventana de datos, todos los parámetros
de costos y exposición, y los dos grupos operados con su pata numeraria y vida media. Si una
cifra del reporte no cuadra con lo que ve en pantalla, lo primero que hay que comparar es el ID
de corrida.

### Vista ejecutiva — para un portfolio manager o un comité de inversión

Responde "¿cuánto ganó, cuánto arriesgó y a dónde se fue el dinero?". Cinco secciones, de arriba
a abajo:

1. **Curva de capital y drawdown.** El panel superior muestra el consolidado con Risk Parity; el
   multiselect agrega los comparativos (pesos iguales, cada grupo solo). El panel inferior es el
   drawdown. El fondo está sombreado en verde, ámbar y rojo según entrenamiento, prueba y
   validación: ahí se ve de un golpe que la curva sube en el tramo verde y baja en los otros dos.
2. **Tarjeta de desempeño.** Las seis métricas que pide el PDF (Sharpe, Sortino, Calmar, Max
   Drawdown, Win Rate, Profit Factor) más capital final, retorno, volatilidad y número de
   operaciones. Los dos selectores eligen **portafolio** y **periodo**. Si elige un periodo fuera
   de muestra con Sharpe negativo, aparece un aviso que remite a las limitaciones del reporte.
3. **Atribución de costos.** Barras apiladas con P&L neto, comisiones y préstamo por partición, y
   los diamantes negros marcan el P&L bruto. La identidad bruto = neto + comisiones + préstamo
   cierra exactamente. Aquí se ve que los costos consumen el 75.2% del P&L bruto global.
4. **Bitácora de operaciones.** Filtrable por **grupo**, **periodo**, **dirección** (spread largo
   o corto) y **resultado** (todas, solo ganadoras, solo perdedoras). Las cuatro métricas de
   arriba se recalculan con el filtro aplicado.
5. **Distribución de retornos.** Histograma del P&L por operación con una normal superpuesta,
   sobre las operaciones **ya filtradas** en la sección anterior. La normal es referencia visual,
   no un supuesto del modelo.

### Vista técnica — para un investigador cuantitativo o un gestor de riesgos

Responde "¿el modelo está sano y la relación de equilibrio sigue ahí?". El selector de **grupo
cointegrado** de arriba cambia todas las gráficas de las secciones 1 a 6; las métricas de
cabecera muestran pata numeraria, vida media, $\beta^\top\alpha$ y rezagos del VECM.

1. **Spread con señales.** $S_t$ y la media latente $\hat\mu_{t|t-1}$, con triángulos para
   entradas largas y cortas y círculos para salidas.
2. **Vector de pesos.** Una línea por activo. La pata numeraria es la constante en 1. Donde una
   línea cruza el cero, esa pata cambió de lado en la relación de equilibrio.
3. **Primer eigenvector de Johansen móvil.** Reestimado en ventanas de 504 días y re-anclado con
   la misma regla de normalización. El eje derecho lleva el estadístico de traza contra su
   crítico: sirve para ver si la cointegración se sostiene a lo largo de la muestra.
4. **Ganancia de Kalman de ambos filtros.** El filtro 1 por componente del estado, el filtro 2
   para la media latente.
5. **z-score con umbrales.** Las seis líneas horizontales son entrada, salida y stop, en ambos
   signos, con los valores que eligió el grid search.
6. **Diagnósticos de convergencia.** Varianza posterior del estado del filtro 1 (escala
   logarítmica, con la marca del fin del burn-in) y, del filtro 2, la incertidumbre de la media
   junto a la sigma adaptativa del spread.
7. **Sharpe móvil de 252 días.** El multiselect permite superponer los cuatro portafolios.

Debajo, el material de **nivel B**: contribuciones al riesgo de Risk Parity contra pesos iguales,
el mapa de calor de correlación entre spreads, la tabla de pesos y la atribución del drawdown
máximo por grupo.

Al final hay tres paneles plegables: el **diagnóstico de los filtros** (con el aviso explícito de
que la autocorrelación no es baja y por qué), la **tabla de los 26 grupos probados** con $m$ y
los falsos positivos esperados, y la comparación de **rebalanceo diario contra banda**.

### Interactividad

Todas las gráficas son de Plotly: zoom arrastrando, doble clic para restablecer, hover con
valores, y clic en la leyenda para ocultar o aislar series.

## Uso de asistencia de IA

Declaración honesta y específica, como exige la sección 5 de los lineamientos.

### Qué hizo la IA

**El código de este repositorio fue generado con Claude Code (Anthropic) bajo nuestra
dirección.** No se trató de autocompletado puntual: la IA escribió los módulos, los docstrings,
las pruebas y la documentación a partir de nuestras instrucciones, y ejecutó el pipeline para
verificar los resultados. Esto aplica a todas las carpetas: `data/`, `cointegration/`,
`kalman/`, `strategy/`, `backtest/`, `performance/`, `dashboard/`, `tests/` y los documentos de
`project_management/`.

La IA también encontró y corrigió varios errores durante el desarrollo, y propuso soluciones
técnicas que aprobamos: la varianza adaptativa del filtro 2, el uso de $\beta^\top\alpha$ en
lugar del $\alpha$ de una sola pata, el cambio al descenso coordinado cíclico en Risk Parity,
y la forma de plantear la prueba 4 ante una autocorrelación que no puede ser baja.

### Qué hicimos nosotros

- **Elegimos el universo de candidatos** entre tres propuestas, y decidimos operar A2 + B1.
- **Aprobamos cada convención** antes de que se implementara: ecuación de observación del
  filtro 1, normalización $\beta^J/\beta^J_{i^*}$, devengo del préstamo (cambiamos la propuesta
  de actual/360 a **actual/365** para que el costo anual fuera exactamente el 0.25% del PDF), e
  interpretación de los "residuales del VECM".
- **Revisamos y aprobamos cada bloque de fases** antes de permitir que avanzara al siguiente.
- **Corregimos decisiones de la IA.** Las correcciones que dimos y que cambiaron el trabajo:
  rechazamos relajar criterios para que A1 o A3 pasaran Johansen (habría sido p-hacking) e
  instruimos agregar B1 declarándolo; corregimos la justificación económica de los bancos
  canadienses, que la IA había descrito como "el mismo instrumento en dos bolsas" cuando son
  cinco bancos distintos que comparten regulador y ciclo de tasas; quitamos la corrección de
  Bonferroni por no haberse visto en clase; exigimos que la prueba 4 reportara la
  autocorrelación de **ambos** filtros sin disfrazarla; prohibimos cambiar cualquier parámetro
  para mejorar los resultados fuera de muestra; y pedimos versionar el snapshot de precios y los
  artifacts por reproducibilidad.
- **Fijamos las prioridades y el calendario** bajo el que se trabajó.

### Qué significa para la evaluación

Somos responsables de todo el código entregado y podemos explicar cualquier línea. Preparamos
la defensa de cada decisión no obvia (valores críticos de MacKinnon, $\beta^\top\alpha$,
varianza adaptativa, selección de rezagos por Ljung-Box, normalización de $\beta$, por qué
fallaron A1 y A3, por qué los resultados fuera de muestra son negativos) con el procedimiento
completo, no solo con el resultado.

No afirmamos haber escrito el código a mano. Afirmamos haberlo dirigido, revisado, corregido y
entendido.

## Reporte

El reporte ejecutivo está en [`docs/reporte.pdf`](docs/reporte.pdf), generado a partir de
[`docs/reporte.md`](docs/reporte.md) con `python docs/build_reporte.py`. Incluye las tres
formulaciones del marco de Powell, el ejemplo numérico trabajado, las nueve figuras obligatorias
y una sección final con las correcciones aplicadas tras una revisión externa.
