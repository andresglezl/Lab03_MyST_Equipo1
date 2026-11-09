# Post-mortem — Lab 03 MyST, Equipo 1

**Borrador con los hechos del proyecto.** Las secciones marcadas con `[REVISAR]` son reflexión
personal que **nosotros debemos escribir**: la rúbrica evalúa esta sección por honestidad y
profundidad analítica, y eso no se delega.

**Ventana del proyecto:** sábado 7 de noviembre de 2026, ~14:30 → domingo 8 de noviembre, 18:00
(~27 h de trabajo efectivo sobre ~33 h de calendario), más una ronda de correcciones posterior a
una revisión externa. Entrega: domingo 8 de noviembre, 23:59. Exposición: lunes 9 de noviembre.

---

## 1. Hitos: qué se cumplió a tiempo y qué se retrasó

| Hito | Objetivo | Resultado | ¿A tiempo? |
|---|---|---|---|
| H1 Pipeline de datos | sáb 14:30 | sáb 14:30 | Sí |
| H2 Cointegración completa | sáb 17:00 | sáb ~16:00 | Sí, adelantado |
| H3 Filtro 1 validado | sáb 19:00 | sáb ~17:30 | Sí, adelantado |
| H4 Filtro 2, VECM y señales | sáb 21:30 | sáb ~21:30 | Sí |
| H5 Backtest y Risk Parity | dom 02:00 | dom ~01:00 | Sí |
| H6 Métricas, figuras y pruebas | dom 11:00 | dom ~10:30 | Sí |
| H7 Dashboard | dom 14:00 | dom ~14:00 | Sí |
| H8 README, Powell y reporte | dom 18:00 | dom ~18:00 | Sí |

Ningún hito se retrasó respecto al plan por horas. Dos factores lo explican, y conviene
distinguirlos porque solo uno es mérito: **(a)** el plan se calendarizó *después* de conocer el
alcance real, con el PDF ya leído y las convenciones acordadas, así que los hitos se fijaron con
información que normalmente no se tiene al planear; **(b)** la regla de priorización declarada
en el charter (simplificar y documentar la limitación antes que refinar) evitó que ningún
bloqueo consumiera tiempo abierto.

`[REVISAR]` — *¿Sienten que el plan por horas fue realista o que se ajustó a lo que ya sabíamos
que podíamos hacer? ¿Hubo momentos en que la presión de tiempo los llevó a aceptar algo que con
más margen habrían hecho distinto?*

---

## 2. Cuellos de botella técnicos

### 2.1 El universo original no cointegró, y eso costó una replanificación

Fue el bloqueo más serio. Ninguno de los 15 grupos de A1 (E&P e integradas) pasó Johansen al 95%:
el mejor estadístico de traza fue 32.06 contra un crítico de 47.85, 16 puntos abajo. Eso activó
la reserva A3 (servicios petroleros), que tampoco pasó (Engle-Granger p = 0.624, ADF del spread
p = 0.107).

El problema no era técnico sino de alcance: el nivel B exige Risk Parity entre **al menos dos
grupos cointegrados disjuntos**, y teníamos uno. Las opciones eran relajar criterios (bajar al
90%, probar `det_order=1`, cambiar la ventana) o ampliar el universo declarándolo.

Decidimos lo segundo. Primero verificamos que el fracaso de A1 era **robusto**: no cambia con
$k_{\text{ar,diff}} \in \{1, 2, 5\}$ ni con $k$ elegido por AIC. Después registramos el pool B1
(bancos canadienses en NYSE) y lo evaluamos con exactamente los mismos umbrales, sin relajar
nada. Pasó RY-TD-BMO-CM. El costo honesto quedó asentado: $m$ subió de 21 a 26 grupos evaluados
y los falsos positivos esperados de 1.05 a 1.3.

**Lo que esto enseñó:** una justificación económica buena no garantiza cointegración. La
correlación de niveles en A1 llegó a 0.79 y de todas formas no cointegraron. Compartir un factor
común (el precio del crudo) es necesario pero no suficiente: cointegrar exige que exista una
combinación lineal con **pesos constantes** cuya desviación sea estacionaria, y las exposiciones
al crudo de XOM (integrada con refinación y química) y de DVN (E&P de shale) tienen distinto
signo y distinta magnitud, y esas magnitudes cambiaron con la revolución del shale.

### 2.2 La varianza constante del filtro 1 no sobrevive el cambio de escala de los precios

La varianza de las innovaciones normalizadas pasa de 0.84 y 0.87 en entrenamiento a **2.14** y
**5.22** en la muestra completa. La causa: $R$ se calibra con precios de 2012–2021 y VLO pasó de ~\$30 a
~\$350. Trabajar en niveles con $R$ fija no aguanta un factor de 10 en la escala.

Se resolvió haciendo que el filtro 2 use varianza de observación **adaptativa**,
$R_{2,t} = \hat\sigma^2_{t|t-1}$, que es literalmente lo que la sección 3.4 pide al definir el
estado como la media **y la varianza** latentes. Manteniendo constante la razón $Q_2/R_2$, la
ganancia estacionaria del filtro queda invariante a la escala. Resultado: la varianza de las
innovaciones normalizadas del filtro 2 se queda en 1.10 y 1.15 sobre la muestra completa, dentro
del rango admisible.

La limitación del filtro 1 **no se corrigió**, se declaró: su diagnóstico se evalúa en
entrenamiento, que es donde su parámetro fue calibrado, y el valor fuera de muestra se reporta en
el README y en `artifacts/performance/filter_diagnostics.csv`.

### 2.3 La prueba 4 pide algo que el modelo no puede cumplir

La sección 3.11.4 exige innovaciones normalizadas con "autocorrelación no significativa", y la
nuestra es $\rho_1 \approx 0.97$–$0.99$ en ambos filtros. No es un error: las innovaciones del
filtro 1 **son** el spread, y el spread es un proceso de reversión a la media con vida media de
~36 días, es decir persistente por construcción. Innovaciones blancas significarían que no hay
reversión que operar.

En lugar de declarar cumplido algo que no se cumple, lo convertimos en una predicción
cuantitativa verificable: un AR(1) con vida media $h$ tiene $\phi = 0.5^{1/h} = 0.9811$ para
$h = 36.3$ d, y la autocorrelación medida es 0.981 y 0.988. La prueba verifica esa coincidencia
con tolerancia de 0.05, y la blancura que sí se exige es la de los residuales del VECM, que es el
modelo que describe la dinámica, y que ambos grupos pasan (Ljung-Box p = 0.278 y 0.057).

`[REVISAR]` — *Esta fue una decisión de criterio, no técnica: elegimos reportar un requisito
incumplido con su explicación en lugar de buscar una métrica que pasara. ¿Están cómodos
defendiendo eso frente al profesor? Vale la pena escribir aquí por qué creen que es la decisión
correcta (o por qué no).*

### 2.4 Cuatro errores encontrados y corregidos durante el desarrollo

Se listan porque son el tipo de cosa que un post-mortem debe registrar, no esconder.

| # | Error | Cómo se detectó | Corrección |
|---|---|---|---|
| 1 | **Estadístico de reversión equivocado.** Se usó el $\alpha$ de la pata numeraria como medida de reversión del VECM. Con eso B1 parecía no revertir ($\alpha_{CM} = +0.0064$, $t = +0.95$) y BMO parecía desestabilizar ($\alpha = +0.0388$, $t = +3.11$). | Al revisar por qué un grupo que pasó Johansen y ADF "no revertía". | La cantidad correcta es $\beta^\top\alpha$, la velocidad de ajuste del propio ECT, que se obtiene premultiplicando el sistema por $\beta^\top$. Resultó −0.0189 y −0.0173, con vidas medias de 36 d consistentes con el ajuste OU independiente. |
| 2 | **Risk Parity con pesos negativos y violación del límite del 80%.** La iteración de punto fijo $w_i \leftarrow 1/(\Sigma w)_i$ no garantiza positividad; con correlación negativa entre spreads (−0.1374) convergió a $w = (-1.109, +2.109)$, y como $\sum|w| = 3.2$ el nocional bruto llegó al **267.9% del capital**. | La exposición bruta máxima impresa en consola era absurda. | Descenso coordinado cíclico, cuya actualización en forma cerrada $w_i = (-b_i + \sqrt{b_i^2 + 4\sigma_i^2/G})/(2\sigma_i^2)$ da $w_i > 0$ por construcción. Resultado: $w = (0.345, 0.655)$, exposición máxima 80.8%. |
| 3 | **Malla de umbrales mal especificada.** Incluía `z_salida = 0.0`, que vuelve la regla de salida inalcanzable (pide $\lvert z\rvert < 0$); el grid search la eligió y B1 quedaba invertido 2,736 de 3,496 días. Además `z_entrada` se topaba en 2.5, el extremo de la malla. | Al revisar por qué un grupo estaba casi siempre en mercado. | Se quitó el 0.0 de la malla de salida y se extendió `z_entrada` hasta 3.0. Ambos óptimos quedaron interiores. |
| 4 | **Fixture de causalidad mal construido.** Recalculaba los cortes 60/20/20 sobre el índice truncado, lo que cambiaba el conjunto de entrenamiento y habría hecho que la prueba comparara dos experimentos distintos en lugar de medir look-ahead. | La prueba falló con un `AttributeError` y al arreglarlo se vio el problema de fondo. | Se pasa la fecha de corte del panel completo como parámetro. |

Un quinto caso merece mención aparte porque **no era un error**: la prueba de normalización
detectó un cambio de signo en $\beta_{PBF}$. Al investigarlo resultó ser un **cruce continuo por
cero**, no un volteo espurio. La prueba se reformuló para distinguir los dos fenómenos, y los
cruces se reportan como resultado. Después se descubrió que no era uno sino **siete** (β_PBF ×1,
β_RY ×2, β_TD ×4): la primera verificación solo había mirado β_PBF.

### 2.5 Herramientas

La máquina de desarrollo no tiene pandoc, LaTeX ni wkhtmltopdf, así que el reporte en PDF se
generó con una cadena propia: Markdown → HTML (`markdown`) → PDF (`PyMuPDF Story`), con las
fórmulas renderizadas a PNG por el mathtext de matplotlib. Funciona y es reproducible, pero es
una solución a la medida de esta máquina.

---

## 3. Cuellos de botella organizacionales

El proyecto se ejecutó en **un solo sprint de ~27 horas**, lo que impuso dos restricciones:

1. **Las fases son estrictamente secuenciales.** Sin cointegración no hay filtro, sin filtro no
   hay señal, sin señal no hay backtest. No había forma de paralelizar entre los dos
   integrantes, así que el reparto de roles del charter funciona como asignación de
   responsabilidad y de autoría, no como trabajo simultáneo.
2. **El modo de trabajo cambió a mitad del proyecto.** El plan original pedía revisar y aprobar
   al terminar **cada fase**. Con el calendario real eso era imposible, así que se pasó a revisar
   solo al cerrar cada **bloque** de fases (F2–F5, F6–F8, F9–F10). Quedó registrado en
   `project_management/plan_fases.md` con la fecha del cambio.

`[REVISAR]` — *El reparto de roles del charter asigna módulos a cada integrante, pero el trabajo
real fue secuencial y dirigido en conjunto. ¿Cómo describirían honestamente la división del
trabajo? ¿Cada uno revisó de verdad los módulos del otro, o eso quedó pendiente?*

`[REVISAR]` — *¿Hubo fricción en la coordinación? ¿Momentos de desacuerdo sobre una decisión
técnica? Si los hubo, vale más contarlos que presentar un proyecto sin roces.*

---

## 4. Qué haríamos distinto en producción

Esta sección lista lo que las limitaciones declaradas implican, no una lista genérica de buenas
prácticas.

1. **Trabajar en logaritmos de precio, o con un $R$ que escale.** La degradación de la varianza
   del filtro 1 (0.96 → 6.11) es consecuencia directa de usar niveles con $R$ constante. En
   producción el filtro 1 también necesitaría varianza adaptativa, o habría que reformularlo en
   logaritmos y aceptar que $\beta$ deja de traducirse directo a número de acciones.
2. **Reestimar el universo y la relación periódicamente, con una regla de desactivación.** Los
   siete cruces por cero de $\beta$ muestran que la relación de equilibrio cambió de estructura
   entre entrenamiento y validación. Un sistema real necesitaría una prueba recurrente de
   cointegración y una regla que **apague** el grupo cuando deje de pasarla, en lugar de operar
   un $\beta$ congelado durante 5 años.
3. **Presupuesto de costos antes de elegir umbrales.** Los costos fueron el 78.6% del P&L bruto
   global y el 160.8% en prueba. Con una comisión de 0.125% por pata y cuatro patas, el edge
   necesario por operación se conoce **antes** de correr nada: optimizar el Sharpe neto fue lo
   correcto, pero debió venir acompañado de un filtro previo que descartara combinaciones cuyo
   turnover implicara un costo mayor al edge esperado.
4. **Validación con walk-forward rodante, no una sola partición 60/20/20.** El sobreajuste de los
   umbrales (Sharpe +0.87 → −0.78) es la firma de optimizar una vez sobre una muestra fija. En
   producción se reestimaría en ventanas rodantes con reoptimización periódica, lo que da muchas
   más observaciones fuera de muestra y una medida más creíble de la degradación.
5. **Universo sin sesgo de supervivencia.** Solo incluimos empresas que siguen listadas hoy; PXD
   y HES fueron adquiridas en 2024 y quedaron fuera. Un estudio serio usaría el universo *tal como
   era* en cada fecha.
6. **Modelar lo que el backtest ignora.** No hay slippage, ni impacto de mercado, ni
   disponibilidad de préstamo, ni límites de concentración del bróker, y se asume que se consigue
   el precio de cierre exacto. Con rebalanceo diario de ocho patas, el slippage sería material.
7. **Automatizar la verificación de reproducibilidad.** El snapshot de precios resolvió el
   problema de que yfinance reajusta la historia, pero lo descubrimos por accidente al ver que
   $\beta_{PBF}$ cambiaba en el sexto decimal entre descargas. En producción eso debería ser una
   prueba, no un hallazgo.

`[REVISAR]` — *De estos siete, ¿cuál creen que habría cambiado más el resultado? Esa es una
pregunta probable en la exposición y conviene tener la respuesta pensada, no improvisada.*

---

## 5. Riesgos del charter: qué se materializó

El charter registró 8 riesgos. Balance real:

| # | Riesgo | ¿Se materializó? | Qué pasó |
|---|---|---|---|
| R1 | Pérdida de cointegración fuera de muestra | **Parcialmente** | B1 pierde en los dos tramos fuera de muestra (−1.93% y −10.82%): ahí la relación sí dejó de funcionar. A2 pierde en prueba (−4.27%) pero **gana en validación** (+4.35%) y termina positivo. El riesgo se materializó en un grupo y no en el otro. |
| R2 | Hallazgo espurio por pruebas múltiples | **Sí** | $m = 26$, $\alpha m = 1.3$ falsos positivos esperados, 2 grupos pasaron. No podemos descartar que uno sea espurio, y el desempeño fuera de muestra es consistente con eso. La mitigación (universo pre-registrado y reportar $m$) no evita el riesgo: lo hace medible y declarable, que es todo lo que se puede pedir. |
| R3 | Divergencia o sobre-reacción del filtro de Kalman | **No** | $\delta = 10^{-3}$ elegido por log-verosimilitud mantuvo el salto diario máximo de $\beta$ en 0.0047. La varianza posterior converge y el burn-in de 60 días resultó suficiente. |
| R4 | Calidad de datos | **No, pero apareció una variante no prevista** | Cero NaN y un solo retorno extremo, que era real (OXY −52% el 2020-03-09). Lo que **no** estaba en el registro: que yfinance reajusta la historia con cada dividendo, lo que rompe la reproducibilidad. Se resolvió congelando el snapshot. |
| R5 | Diversificación aparente entre grupos | **Parcialmente** | La correlación entre spreads es −0.0857 y el consolidado amortiguó el peor caso (−11.16% contra −15.01% de B1 solo). Pero el drawdown máximo no fue un evento conjunto: B1 lo originó en un **86.7%**. |
| R6 | Los costos se comen el edge | **Sí** | Los costos consumen el **75.2%** del P&L bruto global. El préstamo resultó irrelevante (\$5,337 contra \$98,000 de comisión, el 5.2%): el costo que manda es la comisión por pata con rebalanceo diario de cuatro patas. |
| R7 | Look-ahead accidental | **No** | La prueba de causalidad verifica en 3 fechas que el pipeline truncado reproduce $z_t$ con tolerancia $10^{-10}$, y una segunda prueba confirma que el artifact que lee el dashboard es el mismo que produce la cadena causal. |
| R8 | Riesgo de calendario | **No para los hitos, sí para la calidad** | Los 8 hitos se cumplieron en tiempo, pero una revisión externa posterior encontró 12 errores, 2 de ellos con efecto en los resultados (sección 8). El calendario no se desbordó; lo que no alcanzó fue el tiempo de verificación cruzada. |

**Riesgo que faltó en el registro:** la posibilidad de que el universo pre-registrado
simplemente no produjera suficientes grupos cointegrados para el nivel B. Es el riesgo que más
replanificación costó y no estaba anotado. Un charter mejor lo habría incluido con su mitigación:
tener un pool adicional pre-registrado desde el inicio, lo que habría evitado tener que agregarlo
después y subir $m$.

---

## 6. Resultado del proyecto, dicho sin adornos

Portafolio consolidado con Risk Parity:

| Partición | Capital final | Retorno | Sharpe | Max DD |
|---|---|---|---|---|
| Entrenamiento | \$1,134,045 | +13.40% | +0.932 | −1.73% |
| Prueba | \$1,104,616 | −2.60% | −0.410 | −5.77% |
| Validación | \$1,034,057 | −6.39% | −0.969 | −7.11% |
| Global | \$1,034,057 | +3.41% | +0.134 | −11.16% |

Y cada grupo operado por separado, que es lo que más informa:

| Grupo | Train | Prueba | Validación | Global |
|---|---|---|---|---|
| VLO-MPC-PSX-PBF | +29.70% | −4.27% | **+4.35%** | **+29.56%** |
| RY-TD-BMO-CM | +6.66% | −1.93% | −10.82% | −6.72% |

Encontramos dos relaciones que pasan todas las pruebas formales de cointegración en
entrenamiento, con mecanismos económicos nombrables. Con $m = 26$ grupos probados y 1.3 falsos
positivos esperados, **no podemos afirmar que las relaciones encontradas sean estructurales**.

Pero el diagnóstico es distinto para cada grupo y mezclarlos sería un error:

1. **A2 (refinación) terminó positivo**, +29.56% global, perdiendo en prueba y ganando en
   validación. Su relación no se rompió, aunque sí se movió.
2. **B1 (bancos) terminó negativo**, −6.72% global, perdiendo en los dos tramos fuera de muestra.
   Aquí la relación sí dejó de funcionar.
3. **Los costos consumen el 75.2% del P&L bruto global.** Lo que la estrategia genera antes de
   costos existe; tres cuartas partes se quedan en el camino.

Un cuarto hecho, estructural: **los cinco cruces por cero de $\beta$** documentan que en los dos
grupos los pesos del equilibrio se movieron materialmente entre entrenamiento y validación.
$\beta_{PBF}$ pasó de +0.595 a −0.043 y $\beta_{RY}$ de −0.338 a −0.611. La relación que Johansen
estimó en 2012–2021 no es la misma que operamos en 2024–2026.

**Lo que no podemos concluir:** que los umbrales estén sobreajustados. Es una hipótesis razonable
pero con dos tramos fuera de muestra y un grupo que degrada en prueba y se recupera en validación
no hay evidencia para sostenerla. Una versión anterior de este post-mortem la daba por
demostrada; se corrigió.

Risk Parity además rindió **menos** que pesos iguales (+3.41% contra +10.27%), porque iguala
contribuciones al riesgo y no al retorno: asignó el 69.4% del capital al grupo menos volátil, que
resultó el de peor desempeño. Cumplió su objetivo declarado (contribuciones iguales con desvío
0.0e+00) y eso no fue suficiente para mejorar el resultado. Con dos grupos, además, Risk Parity
coincide exactamente con el inverso de la volatilidad: la correlación no interviene.

`[REVISAR]` — *El laboratorio pedía poder sostener si la relación es estructural o un artefacto.
Nuestra respuesta es "no podemos afirmar que sea estructural". Escriban aquí qué aprendieron de
llegar a un resultado negativo después de hacer todo el procedimiento bien: ¿cambia su intuición
sobre el arbitraje estadístico? ¿Qué les sorprendió más?*

`[REVISAR]` — *¿Qué parte del trabajo les costó más entender de verdad, más allá de hacerla
funcionar? Esa es la pregunta que el profesor va a encontrar si hace preguntas dirigidas.*

---

---

## 8. Revisión externa: lo que encontró y lo que costó

El trabajo se sometió a una revisión externa al cerrar el bloque 3. Encontró **doce errores**,
dos de ellos con efecto sobre los resultados. Se documentan aquí con el mismo criterio que el
resto del post-mortem: lo que importa no es que hubiera errores, sino cuánto tardamos en
encontrarlos nosotros y por qué no lo hicimos.

### Los dos críticos

**C1. Dos contabilidades conviviendo.** El grid search y la calibración de la banda usaban un
motor en "unidades de nocional bruto" que definía el peso de cada pata como
$w_j = \beta_j/\sum_k|\beta_k|$, sin precios. Como $\beta$ está en acciones, el peso correcto en
dólares es $w_j = \beta_j P_j/\sum_k|\beta_k P_k|$. El portafolio consolidado sí usaba acciones y
dólares, así que **calibrábamos sobre un portafolio y operábamos otro**: la correlación entre los
retornos diarios de las dos contabilidades era de 0.79 para el grupo bancario.

Lo que duele de este error es que el diseño original tenía la intención correcta. El docstring
del motor decía textualmente que usar el mismo código para calibrar y para reportar evita "la
fuente clásica de discrepancias entre lo calibrado y lo reportado". El problema es que había dos
motores y no uno, y nunca verificamos que coincidieran. **La lección: una afirmación de diseño en
un comentario no es una verificación.** Si hubiéramos escrito una prueba que comparara los
retornos de los dos caminos, habría fallado desde el primer día.

**C2. El filtro 1 no arrancaba en el vector de Johansen.** Se inicializaba con una solución OLS.
El reporte afirmaba que el spread operado era el término de corrección de error del VECM, y con
esa inicialización era falso: el peso de PBF arrancaba en +0.256 contra +0.595 de Johansen.

Este error es de otro tipo. No es un descuido de implementación: es que **escribimos en el
reporte una propiedad que nunca comprobamos**. La frase "el ECT es exactamente el spread
operado" sonaba bien y era coherente con el resto del diseño, pero nadie la contrastó contra el
artifact. Un `assert` de una línea comparando el día 0 del filtro con el vector de Johansen la
habría detectado.

### Los diez menores

Cuatro de ellos son del mismo tipo y conviene verlo junto: **métricas que se veían razonables y
no lo eran.** El Sortino dividía entre la desviación del subconjunto de días negativos en lugar
de la desviación a la baja estándar. El win rate de pesos iguales usaba la bitácora de Risk
Parity. Se publicaba "costos = 160.8% del bruto" cuando el bruto era negativo, que no significa
nada. Y la prueba de contabilidad verificaba una identidad que se cumplía por construcción,
porque el efectivo se definía como capital menos patas.

Los otros seis son de presentación y de redacción: la figura 4 con picos de ±1000, las figuras
que salían como miniaturas en el PDF, `\mathrm{}` con espacios pegados, el dashboard mostrando
"new text" y `$K_t$` literal, la descripción incorrecta de `det_order=0`, y la afirmación de que
el inverso de la volatilidad no resuelve Risk Parity (con dos grupos sí lo resuelve, exactamente).

### Efecto en los resultados

Los umbrales cambiaron en los dos grupos y el resultado se movió. Lo más importante: con el motor
corregido, **el grupo de refinación gana en validación (+4.35%) en lugar de perder**, y termina
en +29.56% global. La conclusión general del trabajo no cambia —no podemos afirmar que las
relaciones sean estructurales— pero sí cambia el diagnóstico por grupo: ya no es cierto que
ninguno de los dos haya sobrevivido fuera de muestra. La tabla completa del antes y después está
en la sección 12 de `docs/reporte.pdf`.

También quitamos del reporte la afirmación de que los umbrales están sobreajustados. Con dos
tramos fuera de muestra y un grupo que degrada en prueba y se recupera en validación, no hay
evidencia para sostenerla. Era una explicación cómoda y la estábamos dando por demostrada.

`[REVISAR]` — *Estos doce errores sobrevivieron a nuestras tres revisiones de bloque. ¿Qué
tendríamos que haber hecho distinto para encontrarlos nosotros? Vale la pena ser concreto: no
"revisar mejor", sino qué verificación específica faltaba.*

`[REVISAR]` — *Dos de los errores consisten en haber escrito en el reporte propiedades que no
verificamos (el ECT, y los dos motores coincidiendo). ¿Cómo evitan eso la próxima vez?*

---

## 7. Lo que sí salió bien

Para que el post-mortem no sea solo una lista de problemas:

- **La disciplina metodológica se mantuvo.** Todo lo que se estima usa solo entrenamiento, y hay
  una prueba automatizada que lo verifica en lugar de suponerlo.
- **Los errores se encontraron con diagnósticos, no por casualidad.** Los cuatro bugs de la
  sección 2.4 salieron de revisar cifras que no cuadraban (exposición de 267.9%, un grupo que no
  revertía, otro siempre en mercado), no de que algo se rompiera.
- **No se maquilló nada.** No se relajó ningún criterio para que A1 pasara, no se cambió ningún
  parámetro para mejorar prueba o validación, y el requisito de la prueba 4 que no se cumple se
  reporta como incumplido con su explicación. Al corregir los errores de la revisión tampoco se
  tocó ningún parámetro para mejorar los resultados: los umbrales nuevos salieron de volver a
  correr la misma búsqueda con el motor corregido.
- **El proyecto es reproducible de verdad.** Snapshot de precios congelado, artifacts
  versionados, semilla fijada, versiones de paquetes registradas en el manifiesto, y el dashboard
  abre en una copia recién clonada sin correr nada.
