# Arbitraje estadístico con cointegración y filtros de Kalman

**Laboratorio 03 — Microestructuras y Sistemas de Trading · ITESO**

**Equipo 1 · Nivel de alcance B**

Rogelio Adrián Arroyo Valencia · Ernesto Andrés González Lomelí

Noviembre de 2026

---

## 1. Resumen ejecutivo

Se construyó una estrategia de arbitraje estadístico sobre portafolios multi-activo de reversión
a la media. De un universo pre-registrado de **26 grupos candidatos** de 4 activos, dos pasaron
el embudo completo de cointegración: **VLO-MPC-PSX-PBF** (refinación estadounidense) y
**RY-TD-BMO-CM** (bancos canadienses listados en NYSE). Ambos se operaron simultáneamente con
capital asignado por Risk Parity.

El resultado del portafolio consolidado es **positivo en entrenamiento y negativo en los dos
tramos fuera de muestra**, y ese contraste es el hallazgo principal del trabajo:

| Partición | Capital final | Retorno | Sharpe | Sortino | Calmar | Max DD | Win rate | Profit factor |
|---|---|---|---|---|---|---|---|---|
| Entrenamiento | 1,134,045 | +13.40% | +0.932 | +1.484 | +0.881 | −1.73% | 76.3% | 3.799 |
| Prueba | 1,104,616 | −2.60% | −0.410 | −0.520 | −0.163 | −5.77% | 42.1% | 0.414 |
| Validación | 1,034,057 | −6.39% | −0.969 | −1.225 | −0.330 | −7.11% | 46.7% | 0.542 |
| Global | 1,034,057 | +3.41% | +0.134 | +0.184 | +0.022 | −11.16% | 61.1% | 1.144 |

Con $m = 26$ grupos probados y $\alpha m = 1.3$ falsos positivos esperados al 5%, **no podemos
afirmar que las relaciones de equilibrio encontradas sean estructurales**.

El diagnóstico difiere entre los dos grupos y conviene no mezclarlos. El grupo de refinación
(VLO-MPC-PSX-PBF) termina **positivo** operado por separado, +29.56% global, con Sharpe +0.601, y
de hecho gana en validación (+4.35%); su mal tramo es prueba (−4.27%). El grupo bancario
(RY-TD-BMO-CM) termina **negativo**, −6.72% global, y pierde en los dos tramos fuera de muestra,
con −10.82% en validación. El resultado agregado del consolidado lo arrastra el segundo.

Los costos pesan: consumen el **75.2%** del P&L bruto global. Y hay evidencia estructural directa
de que la relación de equilibrio se movió: **cinco cruces por cero** en las componentes de los
vectores de pesos, con $\beta_{PBF}$ pasando de +0.595 a −0.043 entre el inicio y el final de la
muestra.

---

## 2. Datos y parámetros

| Parámetro | Valor | Nota |
|---|---|---|
| Fuente | yfinance, cierre ajustado | Snapshot congelado y versionado |
| Ventana | 2012-12-13 a 2026-11-06 | 13.90 años, 3,496 observaciones |
| Activos por grupo | 4 | $2n$ con $n = 2$ integrantes |
| División | 60 / 20 / 20 cronológica | 2,097 / 699 / 700 días |
| Comisión | 0.125% | Por operación, por pata, entrada y salida |
| Tasa de préstamo | 0.25% anual | actual/365 sobre días calendario |
| Anualización de métricas | 252 días hábiles | Distinta del devengo, deliberadamente |
| Capital inicial | 1,000,000 USD | |
| Exposición máxima | 80% del capital consolidado | Nocional bruto |
| Semilla | 42 | `config.SEMILLA` |

La fecha de inicio la impone PBF, el ticker más joven del universo (cotiza desde 2012-12-13). Se
usa **una sola ventana común** para los 20 tickers, no una por grupo, por dos razones: todos los
grupos candidatos se prueban sobre la misma muestra, de modo que sus estadísticos de Johansen son
comparables y la selección no premia al grupo con más historia; y el portafolio consolidado
necesita una única rejilla de fechas alineada.

**Reproducibilidad.** yfinance recalcula el cierre ajustado hacia atrás con cada dividendo o
split: el factor de ajuste de toda la historia previa cambia. Lo comprobamos en este propio
trabajo, donde entre dos descargas separadas por horas el vector cointegrante pasó de
$\beta_{PBF} = 0.594517$ a $0.594516$. Por eso la descarga se congeló una sola vez en
`data/precios_snapshot.parquet`, que se versiona en el repositorio y es lo que `python main.py`
lee por defecto.

Se trabaja con **niveles de precio**, no logaritmos, para que el vector de pesos se traduzca
directamente a número de acciones y la contabilidad multi-pata cierre al centavo. La
consecuencia, que se discute en la sección 10, es que una varianza de observación constante no
sobrevive un cambio de escala de 10× en los precios.

### Figura 1 — Series de precios individuales

![Series de precios individuales de los activos de cada grupo](fig01_precios.png)

---

## 3. Selección del grupo de activos

### 3.1 Universo pre-registrado

El universo se congela antes de calcular cualquier estadístico. Es la defensa frente al problema
de pruebas múltiples, y el campo `momento_registro` de cada pool deja asentado cuándo entró.

| Pool | Tickers | Tendencia estocástica común | Registro |
|---|---|---|---|
| A1 E&P / integradas | XOM, CVX, COP, OXY, EOG, DVN | Nivel del crudo (WTI / Brent) | Original |
| A2 Refinación | VLO, MPC, PSX, PBF, DINO | Margen de refinación (crack 3-2-1) | Original |
| A3 Servicios | SLB, HAL, NOV, RIG | Ciclo global de capex upstream | Reserva, activada al fallar A1 |
| B1 Bancos canadienses | RY, TD, BNS, BMO, CM | Ciclo del Banco de Canadá y mercado hipotecario doméstico | Después de que A1 y A3 fallaran |

**A1 y A2** se eligieron porque están en lados opuestos del mismo insumo: un alza del crudo es
ingreso para una E&P y costo para una refinadora. **B1** se registró después, y la justificación
hay que decirla con precisión: son **cinco bancos distintos**, no el mismo instrumento en dos
bolsas. Cada uno está cross-listed individualmente en NYSE y TSX, pero entre ellos no comparten
flujos de efectivo ni derechos sobre los mismos activos. Lo que comparten es el entorno: el mismo
regulador prudencial (OSFI) con las mismas reglas de capital, el mismo mercado hipotecario
doméstico concentrado y con seguro público (CMHC), y el mismo ciclo de política monetaria del
Banco de Canadá que fija su margen financiero. Ese mecanismo es nombrable y falsable: si cambiara
el régimen regulatorio o el ciclo de tasas se divorciara entre ellos, la relación debería
romperse.

### 3.2 El embudo, paso por paso

**Paso 1, correlación móvil.** Para cada par del grupo se calcula la correlación móvil de 252
días sobre **niveles** de precio y se toma su mediana en el tiempo; el grupo se resume por el
mínimo de esas medianas, es decir el par más débil manda. Se exige mínimo 0.70. Se usan niveles y
no retornos porque el objetivo es detectar una tendencia estocástica común, que es una propiedad
de los niveles. Es un filtro heurístico: la correlación entre series con tendencia está sesgada al
alza, así que no decide nada por sí sola; su función es reducir el número de grupos que llegan a
las pruebas formales.

**Paso 2, Engle-Granger.** Se corre la regresión

$$P_{i,t} = c + \sum_{j \neq i} \theta_j P_{j,t} + u_t$$

y se prueba la estacionariedad del residual. **Un punto que importa:** los valores críticos
tabulados del ADF no son válidos sobre residuales *estimados*, porque el OLS del primer paso
eligió los $\hat\theta_j$ justamente para minimizar la varianza de ese residual, lo que lo hace
parecer más estacionario de lo que es e infla los rechazos. Se usan los valores críticos de
**MacKinnon**, que corrigen por regresores estimados, a través de
`statsmodels.tsa.stattools.coint`. Se prueban las cuatro elecciones posibles de variable
dependiente y se reporta el p-value mínimo, lo cual es apropiado para un pre-filtro cuyo único
fin es descartar grupos sin ninguna evidencia.

**Paso 3, Johansen.** Es el criterio definitivo porque no exige elegir variable dependiente y
detecta más de una relación de cointegración. Especificación: `det_order = 0` y `k_ar_diff = 1`.
En statsmodels, `det_order = 0` mete una constante **no restringida**: entra al VAR pero no se
restringe a la relación de cointegración, de modo que el spread $\beta^\top P$ no se fuerza a
tener media cero. Su nivel lo absorben el intercepto $c_t$ del filtro 1 y la media latente
$\mu_t$ del filtro 2. `k_ar_diff = 1` se elige por parsimonia. El rango se determina con el estadístico de traza por el
procedimiento secuencial, al 95%.

**Paso 4, validación del vector reversivo.** Cuando el rango es mayor que 1 se construye el
spread de cada columna cointegrante, se evalúa ADF y vida media, y se selecciona empíricamente:
criterio primario el p-value del ADF, desempate por vida media más corta. Aquí sí se usa el ADF
estándar, y es legítimo: en este punto $\beta$ ya viene dado por Johansen y no se re-estima, así
que $S_t = \beta^\top P_t$ es una serie observada. La vida media se estima ajustando la versión
discreta de un proceso de Ornstein-Uhlenbeck,

$$\Delta S_t = a + b\,S_{t-1} + e_t, \qquad h = \frac{-\ln 2}{\ln(1+b)}$$

y se exige que caiga en $[5, 126]$ días hábiles: por debajo de 5 los costos se comen la reversión,
por encima de 126 una posición abierta consume capital demasiado tiempo.

### 3.3 Resultados del embudo

Se evaluaron **los 26 grupos** y cada uno se etiquetó con la etapa en que fue descartado. La
tabla completa está en `artifacts/selection/groups_tested.csv`.

| Pool | Grupos | Resultado |
|---|---|---|
| A1 | 15 | **Ninguno pasó.** 9 cayeron en correlación, 6 en Engle-Granger. Traza máxima 32.06 contra crítico 47.85 |
| A2 | 5 | **VLO-MPC-PSX-PBF**: corr 0.714, EG p = 0.0323, traza 54.78 > 47.85, rango 1, ADF p = 0.0004, vida media 37.3 d |
| A3 | 1 | No pasó. EG p = 0.624, traza 40.61, ADF p = 0.107 |
| B1 | 5 | **RY-TD-BMO-CM**: corr 0.927, EG p = 0.0252, traza 54.21 > 47.85, rango 1, ADF p = 0.0056, vida media 33.1 d |

Vectores normalizados de los grupos operados:

| Grupo | Pata numeraria | $\beta$ |
|---|---|---|
| VLO-MPC-PSX-PBF | VLO | $(1,\; -0.787,\; -0.884,\; +0.595)$ |
| RY-TD-BMO-CM | CM | $(-0.338,\; +0.562,\; -0.559,\; 1)$ |

Son disjuntos en activos, condición que exige el nivel B y que el código verifica.

### 3.4 Por qué falló A1, y por qué no relajamos nada

Nuestra hipótesis era que las seis empresas de A1 comparten el nivel del crudo como factor
común. Compartir un factor es **necesario pero no suficiente** para cointegrar: cointegrar exige
que exista una combinación lineal con **pesos constantes** cuya desviación sea estacionaria. Tres
razones económicas lo rompen en A1. Primero, no son el mismo negocio: XOM y CVX son integradas
con refinación y química, cuyo resultado depende también del margen de refinación, que se mueve
*contra* el crudo, mientras COP, EOG y DVN son E&P puras. Segundo, la revolución del shale cambió
las estructuras de costo de forma desigual, lo que es un cambio estructural y no una desviación
estacionaria. Tercero, hay eventos idiosincráticos grandes: OXY cayó −52% el 2020-03-09.

Antes de ampliar el universo verificamos que el fracaso de A1 es **robusto**: no cambia con
$k_{ar,diff} \in \{1, 2, 5\}$ ni con $k$ elegido por AIC. Decidimos no relajar el nivel de
significancia, no probar otras especificaciones determinísticas y no cambiar la ventana. Habría
sido buscar la especificación que hace pasar la prueba en lugar de aceptar el resultado.

### 3.5 Defensa frente al problema de pruebas múltiples

Con $\alpha = 0.05$ y $m$ hipótesis independientes, el número esperado de falsos positivos bajo
la hipótesis nula es

$$\mathbb{E}[\mathrm{FP}] = \alpha \cdot m$$

| Escenario | $m$ | $\mathbb{E}[\mathrm{FP}]$ |
|---|---|---|
| Nuestro universo pre-registrado | 26 | **1.3** |
| Grupos de 4 activos en el S&P 500 | $\binom{500}{4} \approx 2.57 \times 10^{9}$ | $\approx 1.3 \times 10^{8}$ |

Pasaron 2 grupos y esperábamos 1.3 espurios. La conclusión honesta es que **no podemos descartar
que uno de los dos sea un hallazgo espurio**. Nuestra defensa no es el p-value: es el mecanismo
económico nombrado para cada grupo y el desempeño fuera de muestra. Agregar B1 después de que A1
fallara subió $m$ de 21 a 26, y ese costo se reporta en lugar de ocultarse. No se aplicaron
correcciones de multiplicidad tipo Bonferroni porque no forman parte de los métodos vistos en
clase.

---

## 4. Filtro de Kalman 1 — Pesos dinámicos del portafolio

### 4.1 Formulación

El vector de pesos no está identificado hasta fijar la escala, así que se elige una **pata
numeraria** $i^*$ y se pasa al lado izquierdo de la ecuación de observación. Con
$\tilde P_t = (P_{j,t})_{j \neq i^*}$ y estado $x_t = (\theta_t, c_t)$:

$$x_t = x_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, Q)$$

$$y_t = P_{i^*,t} = H_t x_t + v_t, \qquad H_t = (\tilde P_t^\top \;\; 1), \qquad v_t \sim \mathcal{N}(0, R)$$

El vector de pesos se reconstruye con $\beta_{i^*,t} = 1$ y $\beta_{j,t} = -\theta_{j,t}$. La
componente numeraria no es variable de estado: se impone, por lo que vale exactamente 1 y no
aproximadamente.

El estado incluye un **intercepto** $c_t$ que absorbe el nivel del equilibrio. Hace falta
porque la especificación de Johansen que usamos (`det_order = 0`, constante no restringida) no
fuerza al spread a tener media cero.

### 4.2 Normalización y signo

$$i^* = \arg\max_i |\beta^J_i|, \qquad \beta \leftarrow \frac{\beta^J}{\beta^J_{i^*}}$$

Dividir por la componente **con su signo**, no por su valor absoluto, resuelve de un golpe las
dos indeterminaciones del eigenvector de Johansen (escala y signo) y deja $\beta_{i^*} = +1$. La
pata numeraria se elige una sola vez sobre entrenamiento y queda congelada; re-elegirla por
ventana haría saltar la escala de todo el vector. Consecuencia práctica: "largo el spread"
significa siempre largo una unidad de la pata numeraria, de modo que el sentido económico de la
señal nunca se invierte.

### 4.3 Inicialización, $Q$, $R$ y burn-in

| Elemento | Valor | Justificación |
|---|---|---|
| Estado inicial $x_0$ | **Vector de Johansen**: $\theta_0 = -\beta^J_j$, y $c_0$ la media en entrenamiento del residual que deja ese vector | El spread que el filtro produce desde el día 1 es el mismo término de corrección de error sobre el que se estimó la cointegración y se calibraron la vida media y los umbrales |
| $\Sigma_0$ | Covarianza del estimador OLS de $\hat\theta$ | Mide cuánta incertidumbre tiene una estimación de estos coeficientes con esta muestra, y fija la escala de $Q$ |
| $R$ | Varianza en entrenamiento del residual de Johansen: **17.56** (A2), **1.13** (B1) | Literalmente, cuánto no explica la relación de equilibrio estimada |
| $Q$ | $\delta \Sigma_0$ con $\delta = 10^{-3}$ | Hereda unidades y correlaciones de $\Sigma_0$; deja un único escalar libre |
| Burn-in | 60 días hábiles | La varianza posterior ya alcanzó su meseta (figura de convergencia en el dashboard) |

Arrancar en Johansen y no en la solución OLS fue una **corrección posterior a una revisión**. Con
la inicialización OLS, el filtro empezaba en un vector distinto del de Johansen (el peso de PBF
salía en +0.256 contra +0.595), de modo que el spread operado no era el ECT que este reporte
afirmaba estar operando. Ahora el día 0 del filtro reproduce el vector de Johansen exactamente.

La parametrización $Q = \delta \Sigma_0$ es la decisión de diseño que más simplifica la
calibración: en lugar de elegir una matriz, se elige un número. $\delta$ es la razón señal-ruido
del estado, es decir cuánto se permite que los pesos se muevan por día en unidades de su propia
incertidumbre inicial, y se elige por **máxima log-verosimilitud del filtro sobre
entrenamiento** en la malla $\{10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}\}$. La log-verosimilitud no es
una técnica añadida: es un subproducto de la propia recursión que se acumula gratis en cada paso.

### 4.4 Salida: el spread

El spread operable es la **innovación**, es decir usa el estado *predicho* y nunca el posterior:

$$S_t = \hat\beta_{t|t-1}^\top P_t - \hat c_{t|t-1} = y_t - H_t \hat x_{t|t-1} = \nu_t$$

Usar el estado posterior equivaldría a usar la observación del día para juzgar la observación del
día, lo que haría parecer el spread más pequeño de lo que es y sesgaría toda la señal.

### Figura 3 — Evolución del vector de pesos

![Evolución del vector de pesos, una línea por activo](fig03_vector_pesos.png)

La línea constante en 1 es la pata numeraria. **Cinco componentes cruzan cero** a lo largo de la
muestra: $\beta_{PBF}$ una vez (2024-02-05), $\beta_{RY}$ dos y $\beta_{TD}$ dos. Todos son
cruces continuos, no volteos de signo espurios: a ambos lados de cada cruce $|\beta| < 0.003$. Un
peso que cruza cero significa que esa pata **cambió de lado** en la relación de equilibrio, lo
que es un cambio de estructura y no una desviación estacionaria.

Las trayectorias completas, de Johansen al final de la muestra:

| Grupo | Pata | $\beta$ inicial (Johansen) | $\beta$ final | Cruces |
|---|---|---|---|---|
| A2 | MPC | −0.787 | −0.544 | 0 |
| A2 | PSX | −0.884 | −0.659 | 0 |
| A2 | PBF | **+0.595** | **−0.043** | 1 |
| B1 | RY | −0.338 | −0.611 | 2 |
| B1 | TD | +0.562 | +0.484 | 2 |
| B1 | BMO | −0.559 | −0.302 | 0 |

El caso de PBF es el más informativo: su peso decae desde +0.595 hasta cruzar cero en febrero de
2024 y terminar en −0.043. La pata pasó de estar comprada junto a VLO a estar prácticamente
fuera de la relación. La lista completa está en `artifacts/kalman/cruces_por_cero_beta.csv`.

### Figura 4 — Primer eigenvector de Johansen móvil

![Valores del primer eigenvector a lo largo del tiempo](fig04_primer_eigenvector.png)

Esta figura no genera señales: es un diagnóstico de estabilidad estructural. El filtro 1 estima
$\beta_t$ de forma recursiva imponiendo una caminata aleatoria; aquí Johansen se reestima desde
cero en ventanas de 504 días, sin imponer nada. Comparar las dos trayectorias responde si el
filtro se mueve porque la relación realmente cambió o porque $Q$ lo deja perseguir ruido. Cada
ventana se re-ancla con la misma regla de normalización; sin ese re-anclaje la figura mostraría
volteos de signo entre ventanas consecutivas que no significan nada.

---

## 5. Filtro de Kalman 2 — Generación de la señal

### 5.1 Formulación

El filtro 2 opera sobre el spread que produce el filtro 1 y estima su media y su varianza
latentes:

$$\mu_t = \mu_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q_{2,t})$$

$$S_t = \mu_t + \epsilon_t, \qquad \epsilon_t \sim \mathcal{N}(0, R_{2,t})$$

con varianza de observación **adaptativa** $R_{2,t} = \hat\sigma^2_{t|t-1}$ y
$Q_{2,t} = \lambda R_{2,t}$ con $\lambda$ constante. La varianza latente se estima por una
recursión exponencial sobre los errores de predicción del propio filtro:

$$\hat\sigma^2_{t|t-1} = \rho\,\hat\sigma^2_{t-1|t-2} + (1-\rho)\,(S_{t-1} - \hat\mu_{t-1|t-2})^2$$

Los índices importan: la varianza con la que se juzga el día $t$ usa el error del día $t-1$.
$S_t$ nunca entra.

### 5.2 Por qué la varianza debe ser adaptativa

Es una corrección que el propio diagnóstico forzó. Con $R$ constante, la varianza de las
innovaciones normalizadas del filtro 1 pasa de 0.84 y 0.87 en entrenamiento a **2.14** y **5.22**
en la muestra completa, porque $R$ se calibra con precios de 2012–2021 y los niveles del universo
se multiplicaron por casi diez (VLO pasó de ~30 a ~350 USD). Una varianza constante en niveles no
sobrevive ese cambio de escala.

Mantener constante la razón $\lambda = Q_2/R_2$ mantiene constante la **ganancia estacionaria**

$$K_\infty = \frac{-\lambda + \sqrt{\lambda^2 + 4\lambda}}{2}$$

de modo que la velocidad de adaptación del filtro, y por tanto el z-score, quedan invariantes a
la escala del spread. El resultado se verifica: la varianza de las innovaciones normalizadas del filtro 2 se queda en
1.10 y 1.15 sobre la muestra completa, dentro del rango $[0.8, 1.25]$, mientras la del filtro 1
se va a 2.14 y 5.22.

### 5.3 Calibración, atada a la vida media

El único parámetro libre no se elige a ojo. Un filtro de nivel local en estado estacionario es
una media exponencial con factor de olvido $(1 - K_\infty)$, cuya vida media es

$$h_{\mathrm{filtro}} = \frac{\ln 0.5}{\ln(1 - K_\infty)}$$

Se fija $h_{\mathrm{filtro}} = 3\,h_{\mathrm{spread}}$ y se despeja $\lambda$ por bisección. El
factor 3 responde a una necesidad concreta: el equilibrio tiene que moverse **más despacio** que
las desviaciones que se quieren operar. Si $\hat\mu_t$ persiguiera al spread a la misma velocidad
a la que éste revierte, el z-score colapsaría a cero y no habría señal. Con una ventana efectiva
de ~109 días contra una reversión de ~36 días, la reversión queda completamente visible y la
media aún puede reacomodarse ante un cambio de régimen real. Resultado:
$\lambda \approx 4 \times 10^{-5}$ y $K_\infty \approx 0.0064$.

### 5.4 Salida: el z-score

$$z_t = \frac{S_t - \hat\mu_{t|t-1}}{\hat\sigma_{t|t-1}}$$

Ambos estadísticos son **previos a la actualización** del filtro con $S_t$.

---

## 6. Marco de análisis de decisión secuencial (Powell)

### 6.1 Filtro 1 como proceso de decisión secuencial

**Proceso de modelado de seis pasos**

1. **Definir el problema.** Los pesos verdaderos de la relación de cointegración no son
   observables y se desplazan conforme cambia la estructura del mercado. Se requiere una
   estimación adaptativa, disponible en tiempo real y que use solo información pasada, para poder
   recalcular el spread en cada paso.
2. **Modelar el estado.** $x_t = (\theta_t, c_t) \in \mathbb{R}^{n}$, con $\theta_t$ los pesos de
   las patas no numerarias y $c_t$ el nivel del equilibrio. Se acompaña de la covarianza
   $\Sigma_t$, que cuantifica la incertidumbre sobre ese estado.
3. **Modelar la transición.** $x_t = x_{t-1} + w_t$ con $w_t \sim \mathcal{N}(0,Q)$: caminata
   aleatoria. Es deliberada y se defiende así: no tenemos un modelo de cómo deberían moverse los
   pesos de cointegración, solo sabemos que se mueven despacio. La caminata aleatoria es la
   transición que menos estructura impone, y toda la velocidad de adaptación queda concentrada en
   un único parámetro, $Q$, que se calibra con datos.
4. **Modelar la observación.** $y_t = P_{i^*,t} = H_t x_t + v_t$ con
   $H_t = (\tilde P_t^\top \;\; 1)$. La matriz de observación cambia cada día porque contiene
   precios: es el canal por el que la información exógena entra al filtro.
5. **Definir la regla de decisión.** El filtro 1 no toma decisiones de trading. Su "decisión" es
   la estimación que entrega: $\hat x_{t|t-1}$ y el spread $S_t = \nu_t$, que es el insumo del
   filtro 2. Es un problema de estimación secuencial puro.
6. **Definir el mecanismo de aprendizaje.** La actualización bayesiana de Kalman:
   $\hat x_{t|t} = \hat x_{t|t-1} + K_t \nu_t$, con $K_t$ la ganancia.

**Cinco elementos del modelo matemático**

| Elemento | Contenido en el filtro 1 |
|---|---|
| Variables de estado | $\hat x_{t|t} = (\hat\theta_t, \hat c_t)$ y su covarianza $\Sigma_{t|t}$ |
| Variables de decisión | La estimación reportada $\hat x_{t|t-1}$ y el spread $S_t$ |
| Información exógena | El vector de precios $P_t$, que llega del mercado y no se controla |
| Función de transición | $\hat x_{t|t-1} = \hat x_{t-1|t-1}$, $\Sigma_{t|t-1} = \Sigma_{t-1|t-1} + Q$, seguido de la actualización |
| Función objetivo | Minimizar el error cuadrático medio de estimación del estado. Equivalentemente, maximizar la log-verosimilitud, que es el criterio con que se eligió $\delta$ |

### 6.2 Filtro 2 como proceso de decisión secuencial

**Proceso de modelado de seis pasos**

1. **Definir el problema.** Para decidir si el spread está lejos de su equilibrio hace falta
   estimar ese equilibrio y la escala de sus fluctuaciones, y hay que hacerlo sin usar
   información futura. Estandarizar con la media y la desviación de toda la muestra sería
   look-ahead.
2. **Modelar el estado.** $\mu_t$, la media latente del spread, y $\sigma_t^2$, su varianza
   latente, más la covarianza $p_t$ de la estimación de $\mu_t$.
3. **Modelar la transición.** $\mu_t = \mu_{t-1} + \eta_t$, caminata aleatoria otra vez, con
   $Q_{2,t}$ proporcional a la varianza estimada. La varianza evoluciona por la recursión
   exponencial sobre los errores de predicción.
4. **Modelar la observación.** $S_t = \mu_t + \epsilon_t$. La observación es el spread realizado
   que produjo el filtro 1: los dos filtros están encadenados, la salida de uno es la observación
   del otro.
5. **Definir la regla de decisión.** La salida es el z-score
   $z_t = (S_t - \hat\mu_{t|t-1})/\hat\sigma_{t|t-1}$, calculado con los estadísticos previos a
   la actualización.
6. **Definir el mecanismo de aprendizaje.** Actualización de Kalman para $\mu$ y recursión
   exponencial para $\sigma^2$, con $\lambda$ atado a la vida media medida en entrenamiento.

**Cinco elementos del modelo matemático**

| Elemento | Contenido en el filtro 2 |
|---|---|
| Variables de estado | $\hat\mu_{t|t}$, $\hat\sigma^2_{t|t-1}$ y $p_{t|t}$ |
| Variables de decisión | El z-score $z_t$ que se entrega a la lógica de señales |
| Información exógena | El spread $S_t$, que viene del filtro 1 y depende de los precios |
| Función de transición | $\hat\mu_{t|t-1} = \hat\mu_{t-1|t-1}$, $p_{t|t-1} = p_{t-1|t-1} + Q_{2,t}$, y la recursión de $\hat\sigma^2$ |
| Función objetivo | Estimar $\mu_t$ con error cuadrático medio mínimo y producir un $z_t$ de varianza unitaria y sin sesgo de look-ahead |

### 6.3 El sistema completo como un único proceso de decisión secuencial

Aquí sí hay decisiones de trading, y el estado debe distinguir sus tres componentes.

**Proceso de modelado de seis pasos**

1. **Definir el problema.** Asignar 1,000,000 USD entre dos portafolios de reversión a la media
   disjuntos, decidiendo cada día si entrar, mantener o salir de cada uno, maximizando el
   desempeño ajustado por riesgo **neto de costos**, con el nocional bruto acotado al 80% del
   capital consolidado.
2. **Modelar el estado.** Tres componentes, que se detallan abajo.
3. **Modelar la transición.** Los precios evolucionan exógenamente; los filtros actualizan sus
   creencias; las posiciones cambian según la regla de decisión; el capital evoluciona con el P&L
   neto.
4. **Modelar la observación.** El vector de precios de cierre de los ocho activos.
5. **Definir la regla de decisión.** La política es una función del z-score y del tiempo en
   posición, descrita en la sección 7.2.
6. **Definir el mecanismo de aprendizaje.** Dos niveles. Dentro de la muestra de operación, los
   filtros de Kalman actualizan sus creencias cada día. Fuera de ella, todos los parámetros
   estructurales ($\beta$, pata numeraria, $Q$, $R$, $\lambda$, umbrales, ancho de banda, pesos
   de Risk Parity) se estimaron **una sola vez** con la partición de entrenamiento y quedaron
   congelados. El sistema no re-aprende en prueba ni en validación, lo que es precisamente lo que
   permite medir honestamente su degradación.

**El estado del sistema completo, en sus tres componentes**

| Componente | Contenido | Por qué pertenece a esta categoría |
|---|---|---|
| **Física** | Capital $C_t$; acciones en posición $q_{g,j,t}$ de cada pata de cada grupo; efectivo; dirección $d_{g,t} \in \{-1,0,+1\}$; días que lleva abierta cada posición | Son cantidades que existen en el mundo, que el agente controla y que persisten si el agente no hace nada |
| **De información** | Precios de cierre $P_t$ de los ocho activos; días calendario transcurridos desde la barra anterior, que determinan el devengo del préstamo | Datos que llegan del mercado, observables sin error y no controlables |
| **De creencia** | $\hat x_{t|t}$ y $\Sigma_{t|t}$ del filtro 1 (la relación de equilibrio en la que creemos); $\hat\mu_{t|t}$, $\hat\sigma^2_t$ y $p_{t|t}$ del filtro 2 (dónde creemos que está el equilibrio y cuán disperso es) | Son distribuciones sobre cantidades no observables. No existen en el mercado: existen en el modelo, y cambiarían si cambiara el modelo |

La distinción no es decorativa. El P&L lo determina la componente física; el look-ahead se cuela
por la de creencia, porque es la única que se *estima*; y la prueba de causalidad verifica
justamente que la componente de creencia en $t$ dependa solo de la información hasta $t$.

**Cinco elementos del modelo matemático**

| Elemento | Contenido |
|---|---|
| Variables de estado | La unión de las tres componentes anteriores |
| Variables de decisión | $d_{g,t} \in \{-1, 0, +1\}$ por grupo, y las acciones objetivo $q_{g,j,t}$ que de ella se derivan vía el sizing |
| Información exógena | $P_t$ y el calendario |
| Función de transición | Precios exógenos; filtros por Kalman; posiciones por la política; y $C_t = C_{t-1} + \mathrm{PnL}_t - \mathrm{comisión}_t - \mathrm{préstamo}_t$ |
| Función objetivo | Maximizar el Sharpe anualizado del retorno neto del portafolio consolidado, sujeto a $\sum_g \sum_j \lvert q_{g,j,t}\rvert P_{j,t} \le 0.80\, C_{t-1}$ |

### 6.4 La ganancia de Kalman: cálculo e interpretación

$$\nu_t = y_t - H_t \hat x_{t|t-1}, \qquad F_t = H_t \Sigma_{t|t-1} H_t^\top + R, \qquad K_t = \frac{\Sigma_{t|t-1} H_t^\top}{F_t}$$

La interpretación es un **cociente de incertidumbres**: el numerador es cuánta duda tiene el
filtro sobre su propio estado, proyectada al espacio de la observación, y el denominador es la
duda total, que suma esa misma incertidumbre más el ruido de observación. Por tanto

$$K_t = \frac{\text{incertidumbre del modelo}}{\text{incertidumbre del modelo} + \text{ruido de la observación}}$$

Si $R$ es grande frente a $\Sigma_{t|t-1}$, la ganancia tiende a cero y el filtro ignora la
observación del día: confía en su creencia previa. Si $R$ es pequeño, la ganancia crece y el
filtro se mueve casi todo lo que la innovación sugiere. La actualización
$\hat x_{t|t} = \hat x_{t|t-1} + K_t \nu_t$ es un promedio ponderado entre creencia previa y
evidencia nueva, con el peso dictado por sus precisiones relativas.

En nuestro caso, la ganancia del filtro 2 converge a $K_\infty \approx 0.0064$: el filtro
incorpora cada día apenas un 0.64% de la sorpresa, lo que equivale a una media exponencial con
vida media de ~109 días. Es deliberadamente lento, por la razón de la sección 5.3.

### Figura 5 — Ganancia de Kalman de ambos filtros

![Ganancia de Kalman de ambos filtros](fig05_ganancias_kalman.png)

### 6.5 Ejemplo numérico trabajado

Las tablas se generan con los datos y los parámetros reales del proyecto, no con números
inventados, y están en `artifacts/kalman/worked_example_*.csv`. Se toman los seis primeros
periodos posteriores al burn-in del grupo VLO-MPC-PSX-PBF.

**Filtro 2**, con $\lambda = 4 \times 10^{-5}$ y $\rho = 0.99383$:

| $t$ | Fecha | $S_t$ | $\hat\mu_{t|t-1}$ | $\hat\sigma_{t|t-1}$ | $\nu_t$ | $z_t$ | $K_t$ | $\hat\mu_{t|t}$ |
|---|---|---|---|---|---|---|---|---|
| 60 | 2013-03-13 | 1.21980 | 5.26825 | 3.27093 | −4.04845 | −1.23770 | 0.01933 | 5.18998 |
| 61 | 2013-03-14 | 1.12190 | 5.18998 | 3.27645 | −4.06808 | −1.24161 | 0.01894 | 5.11293 |
| 62 | 2013-03-15 | 2.29173 | 5.11293 | 3.28208 | −2.82119 | −0.85958 | 0.01857 | 5.06055 |
| 63 | 2013-03-18 | 1.67289 | 5.06055 | 3.27936 | −3.38766 | −1.03303 | 0.01830 | 4.99857 |
| 64 | 2013-03-19 | 1.63522 | 4.99857 | 3.28006 | −3.36335 | −1.02539 | 0.01800 | 4.93803 |
| 65 | 2013-03-20 | 0.42613 | 4.93803 | 3.28059 | −4.51190 | −1.37533 | 0.01771 | 4.85811 |

Verificación aritmética del periodo 60, que se puede seguir a mano:

- Innovación: $\nu_{60} = S_{60} - \hat\mu_{60|59} = 1.21980 - 5.26825 = -4.04845$.
- z-score: $z_{60} = \nu_{60}/\hat\sigma_{60|59} = -4.04845/3.27093 = -1.23770$. Ambos insumos
  son previos a la actualización.
- Actualización: $\hat\mu_{60|60} = 5.26825 + 0.01933 \times (-4.04845) = 5.18998$.

Lo que se observa en la evolución: la media latente desciende monótonamente de 5.268 a 4.858
porque el spread lleva varios días por debajo de ella, pero lo hace **despacio**, en pasos de
~0.07 por día. Eso es el $K_\infty$ pequeño en acción: el filtro no persigue al spread, y por eso
el z-score se mantiene informativo en torno a −0.9 a −1.4 en lugar de colapsar a cero. La
desviación $\hat\sigma$ se mueve apenas, de 3.271 a 3.281, que es la varianza adaptativa
respondiendo a errores de predicción de tamaño parecido entre sí.

El filtro 1 se tabula en `artifacts/kalman/worked_example_filtro1_*.csv`, con el estado predicho,
la fila de observación $H_t$, la innovación, su varianza, la ganancia por componente y el estado
posterior de cada periodo.

---

## 7. VECM y lógica de señales

### 7.1 El VECM y la velocidad de reversión

Se estima el VECM por OLS, ecuación por ecuación, con $\beta$ **fijo** al vector de Johansen:

$$\Delta P_t = a + \alpha\,\mathrm{ECT}_{t-1} + \sum_{l=1}^{k}\Gamma_l \Delta P_{t-l} + e_t, \qquad \mathrm{ECT}_t = \beta^\top P_t$$

Estimarlo así tiene dos ventajas sobre dejar que la rutina reestime su propio $\beta$: el término
de corrección de error que entra al VECM es exactamente el mismo spread que se opera, y los
$\alpha$ son directamente interpretables.

**La cantidad relevante no es el $\alpha$ de una pata, sino $\beta^\top\alpha$.** Premultiplicando
el sistema por $\beta^\top$ y usando $\mathrm{ECT}_t = \beta^\top P_t$:

$$\Delta \mathrm{ECT}_t = (\beta^\top\alpha)\,\mathrm{ECT}_{t-1} + (\text{términos en diferencias}) + \beta^\top e_t$$

es decir, el spread sigue un AR(1) con coeficiente de ajuste $\beta^\top\alpha$. Se requiere
$\beta^\top\alpha < 0$ para que revierta, y la vida media es
$h = -\ln 2 / \ln(1 + \beta^\top\alpha)$.

Esto corrige un error que cometimos: con el $\alpha$ de la pata numeraria, el grupo B1 parecía no
revertir ($\alpha_{CM} = +0.0064$, $t = +0.95$) y BMO parecía desestabilizar ($\alpha = +0.0388$,
$t = +3.11$). Pero el $\beta$ de BMO es negativo (−0.559), así que su producto estabiliza:

$$\beta^\top\alpha = (-0.338)(0.0205) + (0.562)(0.0050) + (-0.559)(0.0388) + (1)(0.0064) = -0.0194$$

| Grupo | $k$ | $\beta^\top\alpha$ | Vida media | Ljung-Box del ECT |
|---|---|---|---|---|
| VLO-MPC-PSX-PBF | 1 | −0.01888 | 36.3 d | p = 0.278 (11 rezagos) |
| RY-TD-BMO-CM | 8 | −0.01731 | 36.1 d | p = 0.057 (18 rezagos) |

Las vidas medias coinciden con las del ajuste independiente de Ornstein-Uhlenbeck (37.3 y 33.1
días), lo que es una verificación cruzada útil.

**Selección de rezagos.** Se elige el menor $k$ de $\{1,2,3,5,8,10\}$ cuyos residuales de la forma
reducida del ECT pasan Ljung-Box al 5%. Un detalle metodológico importante: la prueba se evalúa
en $k + 10$ rezagos, es decir **más allá** de los rezagos ajustados. Evaluarla dentro es circular,
porque la regresión absorbe por construcción justamente esas autocorrelaciones; con $k = 10$
obteníamos p = 1.0000, que es el artefacto.

### 7.2 De dónde sale el z-score de señal

La sección 3.6 de los lineamientos pide que el z-score se derive de "los residuales del VECM,
normalizados con las estimaciones en tiempo real del filtro 2". **Nuestra interpretación
declarada** es que el residual relevante es el **término de corrección de error**
$\mathrm{ECT}_t = \beta^\top P_t$, es decir la desviación del equilibrio, y no el residual de
pronóstico a un paso de las ecuaciones en diferencias.

Es consistente con la sección 3.4, que es la que define el filtro 2: ahí se establece que el
filtro 2 observa "el spread realizado $S_t$" y produce
$z_t = (S_t - \hat\mu_t)/\hat\sigma_t$ con estadísticos previos a la actualización.

El ECT **es** el spread realizado, con una precisión que conviene hacer explícita: el filtro 1
arranca exactamente en el vector de Johansen, así que en $t=0$ su spread coincide con el ECT del
VECM; a partir de ahí $\hat\beta_t$ evoluciona y lo que se opera es **el ECT con el vector
variante en el tiempo**, no con el vector congelado. Esa es la diferencia entre este trabajo y
un VECM estático, y es deliberada. En una versión anterior de este reporte la afirmación era
falsa porque el filtro se inicializaba en una solución OLS distinta de Johansen; se corrigió tras
una revisión. Si en cambio alimentáramos al filtro 2 con el residual de pronóstico de $\Delta P$,
el z-score mediría cuánto se movió hoy el precio frente a lo pronosticado, que es una sorpresa de
un día y no una desviación del equilibrio, y la regla de trading de la propia sección 3.6 —entrar
cuando el portafolio está por debajo de su equilibrio— dejaría de tener sentido.

El VECM no queda de adorno: aporta la verificación de que $\beta^\top\alpha < 0$, la vida media
implícita que calibra el filtro 2 y el periodo máximo de tenencia, y el diagnóstico de blancura
de sus residuales, que es la prueba de que el modelo dinámico es adecuado.

### 7.3 Reglas de operación

| Regla | Condición |
|---|---|
| Entrada larga en el spread | $z_t < -z_{\mathrm{entrada}}$ estando plano |
| Entrada corta en el spread | $z_t > +z_{\mathrm{entrada}}$ estando plano |
| Salida por reversión | $\lvert z_t \rvert < z_{\mathrm{salida}}$ |
| Stop | $\lvert z_t \rvert > z_{\mathrm{stop}}$ |
| Tenencia máxima | Días en posición $> 2\,h$ |

El stop y la tenencia máxima atacan el modo de pérdida real de esta estrategia, que no es una
sucesión de pérdidas chicas sino una posición que deja de revertir.

Los umbrales se eligen por **grid search sobre la partición de entrenamiento**, con 171
combinaciones coherentes, maximizando el **Sharpe neto de costos**. Se usa el neto y no el bruto
porque una malla optimizada sobre el bruto elegiría siempre el umbral de entrada más bajo, y la
comisión se lo comería fuera de muestra. Se exige un mínimo de 10 operaciones para que el Sharpe
sea interpretable.

| Grupo | $z_{\mathrm{entrada}}$ | $z_{\mathrm{salida}}$ | $z_{\mathrm{stop}}$ | Tenencia máxima | Banda | Sharpe en train | Operaciones |
|---|---|---|---|---|---|---|---|
| VLO-MPC-PSX-PBF | 2.25 | 0.25 | 3.00 | 73 d | 0.075 | +0.928 | 13 |
| RY-TD-BMO-CM | 1.25 | 0.50 | 3.50 | 72 d | 0.100 | +0.429 | 25 |

De las 171 combinaciones, 98 y 79 superan el mínimo de 10 operaciones y son elegibles. Ambos
óptimos son interiores a la malla.

Dos defectos de la malla original se corrigieron al verlos en los resultados: incluía
$z_{\mathrm{salida}} = 0$, que vuelve la regla de salida inalcanzable porque pide
$\lvert z\rvert < 0$, y el grid search la eligió, dejando a un grupo invertido casi toda la
muestra; y $z_{\mathrm{entrada}}$ se topaba en el extremo de la malla.

**Una corrección de fondo, posterior a una revisión externa.** El grid search evaluaba las
combinaciones con una contabilidad en "unidades de nocional bruto" que definía el peso de cada
pata como $w_j = \beta_j / \sum_k|\beta_k|$. Eso ignora los precios, y $\beta$ está en acciones:
el peso correcto en dólares es $w_j = \beta_j P_j / \sum_k|\beta_k P_k|$. Las dos contabilidades
describen portafolios distintos (la correlación entre sus retornos diarios era de 0.79 para el
grupo bancario), así que los umbrales se calibraban sobre un portafolio y se operaban sobre otro.
Ahora el grid search, la calibración de la banda y el resultado final usan **el mismo motor en
dólares**, restringido a entrenamiento para las calibraciones. Los umbrales de la tabla son los
que salieron de volver a correr la búsqueda con el motor corregido.

**Convención de ejecución declarada:** $z_t$ se calcula con precios de cierre del día $t$ y la
operación se ejecuta al cierre del **mismo** día, como una orden market-on-close. No hay
look-ahead, porque $z_t$ solo usa información hasta $t$; la simplificación que sí se asume es que
se consigue el precio de cierre exacto, sin slippage.

### 7.4 Rebalanceo diario contra banda de no-rebalanceo

La política que opera la estrategia es el **rebalanceo diario**, como pide la sección 3.6. La
banda se simula en paralelo para cuantificar su costo relativo, y su ancho se calibra solo con
entrenamiento para que la comparación sea justa.

Ambas políticas se simulan con el mismo motor en dólares, con el grupo operado por separado, de
modo que la comparación aísle el efecto de la política. Turnover y comisión en USD:

| Grupo | Política | Partición | Turnover | Comisión | Días con operación | P&L neto | Sharpe |
|---|---|---|---|---|---|---|---|
| A2 | Diario | Entrenamiento | 29,812,847 | 37,266 | 329 | +296,994 | +0.93 |
| A2 | Banda 0.075 | Entrenamiento | 24,884,480 | 31,106 | 30 | +298,985 | +0.95 |
| A2 | Diario | Prueba | 15,320,260 | 19,150 | 130 | −55,423 | −0.49 |
| A2 | Banda 0.075 | Prueba | 13,658,264 | 17,073 | 16 | −56,530 | −0.50 |
| A2 | Diario | Validación | 16,633,513 | 20,792 | 100 | +53,989 | +0.64 |
| A2 | Banda 0.075 | Validación | 15,734,042 | 19,668 | 15 | +52,915 | +0.60 |
| B1 | Diario | Entrenamiento | 46,361,332 | 57,952 | 818 | +66,629 | +0.43 |
| B1 | Banda 0.100 | Entrenamiento | 42,485,187 | 53,106 | 51 | +78,168 | +0.49 |
| B1 | Diario | Prueba | 22,737,724 | 28,422 | 354 | −20,616 | −0.23 |
| B1 | Banda 0.100 | Prueba | 19,764,883 | 24,706 | 23 | −11,342 | −0.12 |
| B1 | Diario | Validación | 15,866,108 | 19,833 | 389 | −113,209 | −1.23 |
| B1 | Banda 0.100 | Validación | 14,028,972 | 17,536 | 19 | −134,590 | −1.34 |

El hallazgo es claro y algo contraintuitivo: la banda reduce los días con operación en un factor
de 8 a 20, pero **el turnover y la comisión bajan solo entre un 8% y un 17%**. La razón es que el
turnover lo domina la apertura y el cierre de posiciones, no el ajuste diario de $\hat\beta_t$.
Fuera de muestra el resultado es mixto: la banda mejora a B1 en prueba y lo empeora en
validación, y empeora marginalmente a A2 en ambos tramos. No hay evidencia de que la banda sea
preferible al rebalanceo diario que pide la sección 3.6.

### Figura 2 — Spread con señales

![Evolución del spread del portafolio con las señales marcadas](fig02_spread_senales.png)

### Figura 6 — z-score con umbrales

![Serie del z-score con los umbrales de entrada, salida y stop](fig06_zscore.png)

---

## 8. Risk Parity entre grupos (nivel B)

### 8.1 Definición y verificación

Con pesos $w$ y covarianza $\Sigma$ de los retornos de los spreads, la volatilidad del portafolio
$\sigma_p = \sqrt{w^\top \Sigma w}$ es homogénea de grado 1, así que por el teorema de Euler se
descompone exactamente:

$$\sigma_p = \sum_i w_i \frac{\partial \sigma_p}{\partial w_i} = \sum_i \frac{w_i (\Sigma w)_i}{\sigma_p}, \qquad \mathrm{RC}_i = \frac{w_i (\Sigma w)_i}{w^\top \Sigma w}$$

Risk Parity pide $\mathrm{RC}_i = 1/G$ para todo grupo.

**Con exactamente dos grupos, Risk Parity coincide con el inverso de la volatilidad.** No es una
aproximación: es una identidad. Pedir $\mathrm{RC}_1 = \mathrm{RC}_2$ equivale a

$$w_1(\sigma_1^2 w_1 + \sigma_{12} w_2) = w_2(\sigma_2^2 w_2 + \sigma_{12} w_1)$$

y el término cruzado $\sigma_{12}w_1w_2$ aparece idéntico en los dos lados, así que se cancela y
queda $w_1^2\sigma_1^2 = w_2^2\sigma_2^2$, es decir

$$w_1 \sigma_1 = w_2 \sigma_2 \quad\Longleftrightarrow\quad w_i \propto 1/\sigma_i$$

**La correlación no interviene.** Nuestros números lo confirman:
$w_1\sigma_1 = 0.0288552$ y $w_2\sigma_2 = 0.0288553$.

Esto deja de valer con tres o más grupos, donde los términos cruzados ya no se cancelan y el
inverso de la volatilidad solo resuelve el problema si todas las correlaciones son iguales. Por
eso el solver implementado es el general, y no la receta $1/\sigma$: con dos grupos da lo mismo,
pero el código queda correcto si el equipo agrega un tercero. En cualquier caso, el resultado se
**verifica numéricamente** en lugar de darse por bueno.

El retorno del spread que alimenta la covarianza es
$r_{g,t} = \Delta S_{g,t} / \sum_k \lvert\beta_{k,t-1}\rvert P_{k,t-1}$, es decir el cambio del
spread por unidad de nocional bruto. Se usa el spread y no el P&L de la estrategia para que los
pesos no dependan de los umbrales del z-score. Todo se estima solo con entrenamiento.

### 8.2 El solver, y por qué importa

La primera implementación usó la iteración de punto fijo $w_i \leftarrow 1/(\Sigma w)_i$
normalizada, que **no garantiza pesos positivos**. Como la correlación entre los dos spreads es
negativa (−0.1374), el producto $(\Sigma w)_i$ se volvió negativo y la iteración convergió a
$w = (-1.109, +2.109)$. Dos consecuencias: deja de ser una asignación de capital interpretable, y
como $\sum_i \lvert w_i \rvert = 3.2$, el nocional bruto desplegado llegó al **267.9% del
capital**, violando el límite del 80%.

Se reemplazó por **descenso coordinado cíclico**. Pedir $\mathrm{RC}_i = 1/G$ con la normalización
$w^\top \Sigma w = 1$ equivale, para cada $i$, a la ecuación cuadrática
$\sigma_i^2 w_i^2 + w_i b_i - 1/G = 0$ con $b_i = \sum_{j \neq i}\sigma_{ij} w_j$, cuya raíz
positiva es

$$w_i = \frac{-b_i + \sqrt{b_i^2 + 4\sigma_i^2/G}}{2\sigma_i^2}$$

La raíz siempre excede $\lvert b_i \rvert$, así que $w_i > 0$ por construcción.

### 8.3 Resultados

| Grupo | Vol. anual del spread | Peso RP | $w_i\sigma_i$ | $\mathrm{RC}$ con RP | $\mathrm{RC}$ con pesos iguales |
|---|---|---|---|---|---|
| VLO-MPC-PSX-PBF | 9.44% | 0.3057 | 0.0288552 | 0.500000 | 0.8604 |
| RY-TD-BMO-CM | 4.16% | 0.6943 | 0.0288553 | 0.500000 | 0.1396 |

Convergencia en 5 iteraciones; desvío máximo contra el objetivo $1/G$: **0.0e+00**, dentro de la
tolerancia declarada de $10^{-6}$. La columna $w_i\sigma_i$ muestra la identidad de la sección
8.1: los dos productos coinciden hasta el séptimo decimal. Risk Parity asigna más capital al
grupo menos volátil, que es la propiedad económica esperada. Con pesos iguales el reparto de
riesgo sería 86% / 14%.

### 8.4 Comparación de asignaciones, y por qué Risk Parity rindió menos

| Portafolio | Capital final | Retorno | Sharpe | Sortino | Max DD |
|---|---|---|---|---|---|
| Consolidado Risk Parity | 1,034,057 | +3.41% | +0.134 | +0.184 | −11.16% |
| Consolidado pesos iguales | 1,102,722 | +10.27% | +0.363 | +0.537 | −9.42% |
| VLO-MPC-PSX-PBF solo | 1,295,560 | +29.56% | +0.601 | +0.972 | −6.90% |
| RY-TD-BMO-CM solo | 932,805 | −6.72% | −0.193 | −0.254 | −15.01% |

Risk Parity quedó **por debajo** de pesos iguales, y las dos por debajo del mejor grupo solo. La
razón no es un defecto del método sino lo que el método promete: iguala contribuciones al
**riesgo**, no al retorno. Asignó el 69.4% del capital al grupo menos volátil, que resultó ser el
de **peor** desempeño. Igualar riesgo sin información sobre el retorno esperado produce
exactamente este resultado cuando la volatilidad y el desempeño no están alineados, y aquí el
efecto es grande porque el grupo bancario tiene menos de la mitad de la volatilidad del de
refinación (4.16% contra 9.44%).

### 8.5 ¿La diversificación es real o aparente?

La correlación entre los spreads es **−0.0857**: baja y negativa, lo que sugiere diversificación
real, con impulsores económicos distintos (margen de refinación estadounidense contra ciclo de
tasas canadiense). Y en lo que la diversificación promete, funcionó: el max drawdown del
consolidado (−11.16%) es mejor que el del peor grupo individual (−15.01%).

Pero la atribución matiza el optimismo. El drawdown máximo del consolidado, de −11.16%, va del
2022-03-16 al 2025-07-28 (843 días hábiles) y se reparte así:

| Grupo | P&L neto en la ventana | Fracción de la caída |
|---|---|---|
| VLO-MPC-PSX-PBF | −17,171 | 13.3% |
| RY-TD-BMO-CM | −111,752 | **86.7%** |

No fue un evento conjunto: lo originó B1 casi en su totalidad. **Una correlación baja entre
spreads no protege de que un grupo simplemente deje de funcionar.** La diversificación amortigua
choques simultáneos, no deterioros estructurales de un componente.

### Figura 9 — Contribuciones al riesgo y correlación entre spreads

![Contribuciones al riesgo por grupo y correlación entre los spreads](fig09_risk_parity.png)

---

## 9. Resultados y análisis cuantitativo

### 9.1 Desempeño del portafolio consolidado

Portafolio consolidado con Risk Parity:

| Partición | Capital inicial | Capital final | Retorno | Sharpe | Sortino | Calmar | Max DD | Win rate | Profit factor |
|---|---|---|---|---|---|---|---|---|---|
| Entrenamiento | 1,000,000 | 1,134,045 | +13.40% | +0.932 | +1.484 | +0.881 | −1.73% | 76.3% | 3.799 |
| Prueba | 1,134,045 | 1,104,616 | −2.60% | −0.410 | −0.520 | −0.163 | −5.77% | 42.1% | 0.414 |
| Validación | 1,104,616 | 1,034,057 | −6.39% | −0.969 | −1.225 | −0.330 | −7.11% | 46.7% | 0.542 |
| Global | 1,000,000 | 1,034,057 | +3.41% | +0.134 | +0.184 | +0.022 | −11.16% | 61.1% | 1.144 |

Cada grupo operado por separado, con todo el capital asignado a él:

| Grupo | Partición | Capital final | Retorno | Sharpe | Max DD | Win rate | Profit factor |
|---|---|---|---|---|---|---|---|
| VLO-MPC-PSX-PBF | Entrenamiento | 1,296,994 | +29.70% | +0.928 | −4.35% | 69.2% | 5.871 |
| VLO-MPC-PSX-PBF | Prueba | 1,241,571 | −4.27% | −0.494 | −6.90% | 14.3% | 0.305 |
| VLO-MPC-PSX-PBF | Validación | 1,295,560 | **+4.35%** | **+0.640** | −2.44% | 57.1% | 3.263 |
| VLO-MPC-PSX-PBF | Global | 1,295,560 | +29.56% | +0.601 | −6.90% | 51.9% | 2.725 |
| RY-TD-BMO-CM | Entrenamiento | 1,066,629 | +6.66% | +0.429 | −3.24% | 80.0% | 2.552 |
| RY-TD-BMO-CM | Prueba | 1,046,014 | −1.93% | −0.225 | −6.78% | 58.3% | 0.437 |
| RY-TD-BMO-CM | Validación | 932,805 | −10.82% | −1.226 | −10.96% | 37.5% | 0.306 |
| RY-TD-BMO-CM | Global | 932,805 | −6.72% | −0.193 | −15.01% | 66.7% | 0.739 |

El desglose por grupo es lo que más informa. **El grupo de refinación gana en validación**
(+4.35%, Sharpe +0.640) después de perder en prueba, y termina positivo en toda la muestra. El
grupo bancario pierde en los dos tramos fuera de muestra, y su −10.82% de validación es lo que
arrastra al consolidado. No se trata, por tanto, de un fracaso uniforme de la metodología: se
trata de dos relaciones con destinos distintos.

### Figura 7 — Curvas de capital por partición

![Curvas de capital para entrenamiento, prueba y validación con drawdown](fig07_curvas_capital.png)

### 9.2 Atribución de costos

La identidad $\mathrm{PnL\ bruto} = \mathrm{PnL\ neto} + \mathrm{comisiones} + \mathrm{préstamo}$
cierra exactamente (residuo 0.0 en todas las particiones).

| Partición | P&L bruto | Comisiones | Préstamo | P&L neto | Costos / bruto | Costos / capital medio |
|---|---|---|---|---|---|---|
| Entrenamiento | 188,635 | 51,809 | 2,781 | 134,045 | 28.9% | 5.08% |
| Prueba | −2,042 | 26,131 | 1,256 | −29,429 | n/a | 2.44% |
| Validación | −49,198 | 20,061 | 1,300 | −70,559 | n/a | 2.04% |
| Global | 137,395 | 98,000 | 5,337 | 34,057 | **75.2%** | **9.58%** |

Tres lecturas. La primera: los costos se llevan el **75.2% del P&L bruto global**. Lo que la
estrategia genera antes de costos existe, pero tres cuartas partes se quedan en el camino.

La segunda es metodológica. El cociente costos / bruto **solo tiene lectura cuando el bruto es
positivo**, porque dice qué fracción de la ganancia se fue en costos. En prueba y validación el
bruto es negativo o casi cero, y el cociente se dispara sin significar nada: con un bruto de
−2,042 y costos de 27,386, la versión anterior de este reporte publicaba un "160.8%" que no
describía nada. Ahí la medida informativa es el costo sobre el capital medio del tramo, que es
2.44% y 2.04% anualizables sobre un capital de ~1.1 millones. En esos tramos el problema no es
que los costos sean grandes, sino que **no hay P&L bruto del que descontarlos**.

La tercera, menos obvia: el **préstamo es irrelevante**, 5,337 USD contra 98,000 de comisión, es
decir el 5.2% de los costos. El costo que manda es la comisión de 0.125% por pata con rebalanceo
diario de cuatro patas por grupo, no el 0.25% anual sobre el corto.

### 9.3 Distribución de resultados por operación

### Figura 8 — Distribución de retornos por operación

![Histograma de P&L por operación con una distribución normal superpuesta](fig08_distribucion_retornos.png)

### 9.4 Diagnóstico de los filtros: lo que se cumple y lo que no

La prueba obligatoria 4 exige innovaciones normalizadas con varianza cercana a uno y
autocorrelación no significativa. Reportamos ambas cosas sin maquillarlas. Las cifras las produce
`pytest` y quedan en `artifacts/performance/filter_diagnostics.csv`.

| Grupo | Filtro | Tramo | Varianza | $\rho_1$ | $\phi = 0.5^{1/h}$ | Ljung-Box(10) |
|---|---|---|---|---|---|---|
| A2 | 1 | Entrenamiento | 0.840 | 0.980 | 0.9811 | 0.0000 |
| A2 | 1 | Muestra completa | **2.140** | 0.981 | 0.9811 | 0.0000 |
| A2 | 2 | Entrenamiento | 1.041 | 0.975 | 0.9811 | 0.0000 |
| A2 | 2 | Muestra completa | 1.102 | 0.974 | 0.9811 | 0.0000 |
| A2 | VECM (ECT) | Entrenamiento | — | — | — | **0.2780** |
| B1 | 1 | Entrenamiento | 0.869 | 0.982 | 0.9810 | 0.0000 |
| B1 | 1 | Muestra completa | **5.223** | 0.988 | 0.9810 | 0.0000 |
| B1 | 2 | Entrenamiento | 1.011 | 0.976 | 0.9810 | 0.0000 |
| B1 | 2 | Muestra completa | 1.147 | 0.975 | 0.9810 | 0.0000 |
| B1 | VECM (ECT) | Entrenamiento | — | — | — | **0.0567** |

**La varianza cercana a uno se cumple, con una salvedad declarada.** El filtro 2 la cumple en toda
la muestra. El filtro 1 la cumple en entrenamiento, que es donde su $R$ fue calibrada, y no fuera
de muestra, por la razón de escala de la sección 5.2.

**La autocorrelación no significativa NO se cumple, y no puede cumplirse.** $\rho_1 \approx 0.97$
a $0.99$ en los dos filtros. No es un defecto de implementación: las innovaciones del filtro 1
**son** el spread, y el spread es un proceso de reversión a la media con vida media de ~36 días,
es decir persistente por construcción. Si estas innovaciones fueran ruido blanco, el spread no
tendría reversión que explotar y la estrategia no tendría razón de existir.

En lugar de declarar cumplido un requisito que no se cumple, lo convertimos en una **predicción
cuantitativa verificable**. Un AR(1) con vida media $h$ tiene coeficiente
$\phi = e^{-\ln 2/h} = 0.5^{1/h}$. Con $h = 36.3$ y $36.1$ días, $\phi = 0.9811$ y $0.9810$, y la
autocorrelación medida coincide dentro de 0.008. La prueba verifica esa coincidencia con tolerancia
declarada de 0.05, de modo que la persistencia queda **explicada por el modelo** y no atribuida a
un error.

El diagnóstico de blancura que sí corresponde exigir es el de los **residuales del VECM**, que es
el modelo que describe la dinámica del spread porque incluye el término de corrección de error y
los rezagos en diferencias. Esos sí deben ser ruido blanco, y lo son.

### 9.5 Pruebas obligatorias

Las cinco pruebas corren con `pytest -q`: **29 pruebas, 29 pasan**.

| # | Prueba | Qué verifica |
|---|---|---|
| 1 | Causalidad | En 3 fechas fuera de entrenamiento, recalcular el pipeline truncado en $t$ reproduce $z_t$ con tolerancia $10^{-10}$. Una segunda prueba confirma que el artifact que lee el dashboard es el mismo que produce la cadena causal |
| 2 | Normalización | $\beta_{i^*,t} = 1$ exacto ($10^{-12}$); todo cambio de signo es un cruce continuo por cero y no un volteo espurio; el salto diario de $\beta$ está acotado; el eigenvector móvil está re-anclado |
| 3 | Contabilidad multi-pata | $C_t = \mathrm{efectivo}_t + \sum$ patas, con el efectivo llevado de forma **independiente** como saldo de caja y reconstruido desde los flujos del libro; la comisión agregada coincide con la reconstruida desde el libro; el préstamo solo se devenga sobre cortos; el nocional bruto nunca excede el 80% de $C_{t-1}$ |
| 4 | Filtros | Varianza de innovaciones normalizadas en $[0.8, 1.25]$; autocorrelación de **ambos** filtros reportada y contrastada contra $\phi$; residuales del VECM blancos; $\beta^\top\alpha < 0$ |
| 4-B | Risk Parity | $\max_g \lvert \mathrm{RC}_g - 1/G \rvert < 10^{-6}$, verificado contra el artifact y recalculado desde la covarianza; pesos positivos que suman 1 |

---

## 10. Limitaciones

Se declaran explícitamente, porque la rúbrica las evalúa y porque son el contexto en que hay que
leer los resultados.

1. **El filtro 1 tiene varianza de observación constante en niveles de precio.** Su diagnóstico
   se degrada fuera de muestra (varianza de innovaciones 2.35 y 6.11 contra el rango
   $[0.8, 1.25]$). Se mitigó haciendo adaptativa la varianza del filtro 2, pero el filtro 1 no se
   corrigió.
2. **La autocorrelación de las innovaciones no es baja** y el requisito correspondiente de la
   prueba 4 no se cumple en su forma literal. Se explica, se cuantifica contra la predicción
   teórica y se declara.
3. **El grid search optimiza sobre una sola muestra.** El procedimiento es metodológicamente
   correcto (solo entrenamiento, sin look-ahead verificado por prueba), pero con dos tramos
   fuera de muestra no hay evidencia suficiente para afirmar ni descartar sobreajuste de los
   umbrales: el grupo de refinación degrada en prueba y se recupera en validación, lo que no es
   el patrón que produciría un sobreajuste simple. Queda como hipótesis no resuelta.
4. **Sesgo de supervivencia en el universo.** Solo se incluyeron empresas que siguen listadas hoy;
   PXD y HES fueron adquiridas en 2024 y quedaron fuera.
5. **El universo se amplió a mitad del análisis.** B1 se registró después de que A1 y A3
   fallaran. Se declara, y $m$ se actualizó de 21 a 26 con el consiguiente aumento de falsos
   positivos esperados.
6. **El modelo de costos omite slippage, impacto de mercado, disponibilidad de préstamo y
   límites de concentración del bróker.** Se asume que se consigue el precio de cierre exacto.
   Con rebalanceo diario de ocho patas, el slippage sería material.
7. **Una sola partición 60/20/20, no walk-forward rodante.** Hay exactamente dos periodos fuera
   de muestra, lo que es poca evidencia para medir degradación.
8. **$\beta$ se congela durante cinco años.** Los siete cruces por cero muestran que la relación
   cambió de estructura; un sistema real necesitaría reestimación periódica y una regla de
   desactivación del grupo.
9. **El límite del 80% se aplica con el capital del cierre previo**, así que la exposición medida
   contra el capital corriente puede llegar a 80.8%. El límite no se viola; la diferencia es de
   convención y se verifica día por día.
10. **Tasa libre de riesgo cero** en Sharpe y Sortino.
11. **Dos tramos fuera de muestra.** Con una sola partición 60/20/20 hay exactamente dos
    observaciones independientes del desempeño fuera de muestra por grupo. Es muy poco para
    distinguir degradación estructural de ruido, y es la razón por la que este reporte evita
    afirmar causas que no puede demostrar.

---

## 11. Conclusiones

**Sobre la pregunta del laboratorio.** Se identificaron dos relaciones de equilibrio que pasan
todas las pruebas formales de cointegración en entrenamiento y que tienen mecanismos económicos
nombrables. Con $m = 26$ grupos probados y 1.3 falsos positivos esperados, **no podemos afirmar
que las relaciones encontradas sean estructurales**: el número de grupos que pasó es del mismo
orden que el número de falsos positivos que esperábamos bajo la hipótesis nula.

**Los dos grupos tuvieron destinos distintos**, y mezclarlos sería el error de lectura más fácil
de cometer:

1. **VLO-MPC-PSX-PBF (refinación) terminó positivo**: +29.56% global operado por separado, con
   Sharpe +0.601, perdiendo en prueba (−4.27%) y ganando en validación (+4.35%). Su relación de
   equilibrio no se rompió, aunque sí se movió: $\beta_{PBF}$ pasó de +0.595 a −0.043.
2. **RY-TD-BMO-CM (bancos) terminó negativo**: −6.72% global, perdiendo en los dos tramos fuera
   de muestra, con −10.82% en validación. Aquí la relación sí dejó de funcionar.

El consolidado queda en +3.41% porque Risk Parity asignó el 69.4% del capital al segundo.

**Evidencia estructural.** Los cinco cruces por cero en las componentes de $\hat\beta_t$
documentan que en los dos grupos los pesos del equilibrio se movieron materialmente entre
entrenamiento y validación. La relación que Johansen estimó sobre 2012–2021 no es la misma que se
operó en 2024–2026, lo cual es exactamente lo que el filtro de Kalman estaba puesto ahí para
seguir, y lo que una estrategia de producción tendría que vigilar con una regla de desactivación.

**Lo que NO podemos concluir.** Que los umbrales estén sobreajustados es una hipótesis razonable
pero no demostrada: con dos tramos fuera de muestra y un grupo que se recupera en validación, no
hay evidencia suficiente. Lo decimos como pregunta abierta y no como causa establecida.

**Sobre los costos.** Con una comisión de 0.125% por pata y rebalanceo diario de cuatro patas por
grupo, los costos consumieron el 75.2% del P&L bruto global. El préstamo resultó irrelevante
(5.2% de los costos). Para una estrategia de este tipo, el presupuesto de costos debería ser una
restricción de diseño y no un resultado del backtest.

**Sobre Risk Parity.** Cumplió exactamente lo que promete: contribuciones al riesgo iguales con
desvío numérico nulo. Y rindió menos que una asignación de pesos iguales, porque iguala riesgo y
no retorno, y asignó la mayor parte del capital al grupo que peor funcionó. La diversificación
entre grupos amortiguó el peor drawdown individual, pero el 87.6% del drawdown del consolidado lo
originó un solo grupo: una correlación baja entre spreads no protege de un deterioro estructural.

**Lo que el trabajo sí demuestra.** Que el procedimiento completo —universo pre-registrado,
embudo de selección documentado, filtros formulados como procesos de decisión secuencial,
calibración sin look-ahead verificada por prueba automatizada, contabilidad multi-pata conciliada
y asignación de capital verificada numéricamente— se puede ejecutar con disciplina y llegar a un
resultado negativo. Distinguir eso de un error de implementación es, probablemente, la parte más
útil del ejercicio.

---

## 12. Correcciones aplicadas tras una revisión externa

El trabajo se sometió a una revisión que encontró errores. Se documentan aquí porque dos de ellos
cambiaron los resultados, y porque las cifras de las versiones anteriores de este reporte ya no
son válidas.

### Críticos, que cambiaron los números

**C1. El grid search calibraba sobre un portafolio distinto del que se operaba.** La función que
convertía $\beta$ en pesos de cartera usaba $w_j = \beta_j / \sum_k|\beta_k|$, ignorando los
precios. Pero $\beta$ está en **acciones**, así que el peso en dólares es
$w_j = \beta_j P_j / \sum_k |\beta_k P_k|$: omitir el precio reparte el capital como si todos los
activos costaran lo mismo. Como el motor consolidado sí trabajaba en acciones y dólares, había
dos contabilidades distintas conviviendo, con una correlación de 0.79 entre sus retornos diarios
para el grupo bancario. Los umbrales y el ancho de banda se calibraban sobre una y el resultado
se medía sobre la otra.

Se unificó: el grid search, la calibración de la banda y la comparación de políticas llaman ahora
al mismo motor en dólares (`walkforward.simular_consolidado` a través de
`engine.simular_grupo_en_dolares`), restringido a entrenamiento para las calibraciones, y la
banda de no-rebalanceo se implementa sobre acciones. El grid search se volvió a correr completo.

**C2. El filtro 1 no arrancaba en el vector de Johansen.** Se inicializaba con una solución OLS,
que es un vector distinto (el peso de PBF salía en +0.256 contra +0.595 de Johansen). Este
reporte afirmaba que el spread operado era el término de corrección de error del VECM, y con esa
inicialización era falso. Ahora $\theta_0 = -\beta^J$, el intercepto es la media en entrenamiento
del residual que deja ese vector, y $R$ es su varianza; $\Sigma_0$ sigue saliendo de la regresión
OLS auxiliar. El día 0 del filtro reproduce el vector de Johansen exactamente.

### Menores

| # | Error | Corrección |
|---|---|---|
| 1 | La figura 4 tenía picos de ±1000 por dividir entre una componente numeraria casi cero | Eje acotado a $[-3, 3]$ y ventanas sin cointegración al 95% sombreadas, en la figura y en el dashboard |
| 2 | El win rate y el profit factor de pesos iguales usaban la bitácora de Risk Parity | Cada portafolio construye su bitácora en dólares con su propio P&L por grupo |
| 3 | El Sortino dividía entre la desviación estándar del subconjunto de días negativos | Desviación a la baja estándar, $\sqrt{\overline{\min(r,0)^2}}$ sobre todos los días |
| 4 | La prueba de contabilidad era trivial: el efectivo se definía como capital menos patas | El motor lleva el efectivo como saldo propio y la prueba lo reconstruye desde los flujos del libro |
| 5 | Se afirmaba que el inverso de la volatilidad no resuelve Risk Parity | Con dos grupos sí lo resuelve exactamente; se demuestra en la sección 8.1 |
| 6 | Se describía `det_order=0` como constante restringida a la relación de cointegración | Es una constante **no restringida**; corregido en la sección 3.2 y en el código |
| 7 | Se publicaba "costos = 160.8% del bruto" con bruto negativo | El cociente solo se reporta con bruto positivo; con bruto negativo se usa el costo sobre capital medio |
| 8 | Se daba el sobreajuste de umbrales por demostrado | Se reporta como hipótesis no resuelta |
| 9 | Las figuras 2, 4, 5 y 7 salían como miniaturas al pie de página | Ancho completo con salto de página; los `\mathrm{}` con espacios ya no se pegan |
| 10 | El dashboard mostraba "new text" en las líneas de umbral negativas y `$K_t$` literal en los ejes | Etiquetas en texto plano y anotación suprimida correctamente |

### Efecto en las cifras

| Magnitud | Antes | Después |
|---|---|---|
| Umbrales A2 (entrada / salida / stop) | 1.50 / 1.00 / 3.00 | **2.25 / 0.25 / 3.00** |
| Umbrales B1 (entrada / salida / stop) | 1.00 / 0.50 / 4.00 | **1.25 / 0.50 / 3.50** |
| Banda A2 / B1 | 0.025 / 0.075 | **0.075 / 0.100** |
| $R$ del filtro 1, A2 / B1 | 14.09 / 0.86 | **17.56 / 1.13** |
| Consolidado RP, capital final | 1,044,245 | **1,034,057** |
| Consolidado RP, Sharpe entrenamiento | +0.869 | **+0.932** |
| Consolidado RP, Sharpe validación | −0.908 | **−0.969** |
| A2 solo, retorno global | +21.43% | **+29.56%** |
| A2 solo, validación | −0.67% | **+4.35%** |
| B1 solo, retorno global | −4.08% | **−6.72%** |
| Costos sobre bruto global | 78.6% | **75.2%** |
| Max drawdown del consolidado | −12.61% | **−11.16%** |
| Cruces por cero de $\beta$ | 7 | **5** |
| Pruebas | 27 | **29** |

El cambio cualitativo más importante: con el motor corregido, **el grupo de refinación gana en
validación** en lugar de perder, y termina claramente positivo en toda la muestra. La conclusión
general no cambia (no podemos afirmar que las relaciones sean estructurales), pero el diagnóstico
por grupo sí: ya no es cierto que ninguno de los dos haya sobrevivido fuera de muestra.
