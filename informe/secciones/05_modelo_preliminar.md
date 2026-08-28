# Modelo preliminar de clasificación

Esta sección cubre la parte del ejercicio 6 que corresponde a la entrega de avance: la
descripción del modelo preliminar. El alcance es deliberado. Lo que se establece aquí es la
**infraestructura de evaluación** —partición congelada, baseline, función de evaluación
reutilizable y métrica de selección— sobre la que la entrega final escalará a más
combinaciones. Todo el trabajo está en `notebooks/04_modelo_preliminar.ipynb` y las métricas
en `results/tables/metricas_preliminares.csv`.

Además de entrenar modelos, esta sección pone a prueba las dos hipótesis que el análisis
exploratorio dejó planteadas. Una se confirma y la otra no, y ambas se reportan.

## Partición de los datos

La partición es estratificada 80/20 con `random_state=42`, la semilla que declara
`src/config.py` e importan todos los cuadernos. Se aplica **después** de la deduplicación.
Ese orden no es un detalle: si las 128 filas duplicadas o contradictorias se eliminaran
después de partir, el mismo tweet podría quedar a ambos lados y el modelo sería evaluado
sobre textos que ya vio, produciendo una métrica optimista por una razón artificial.

| Conjunto | Tweets | No desastre | Desastre |
|---|---|---|---|
| Entrenamiento | 5,988 | 57.41% | 42.59% |
| Prueba | 1,497 | 57.38% | 42.62% |

Las proporciones coinciden hasta la segunda cifra decimal, lo que confirma que la
estratificación funcionó y que el conjunto de prueba es representativo del corpus.

La partición se congela en `data/processed/particion.parquet`. El motivo es concreto: el
ejercicio 10 de la entrega final vuelve a entrenar el clasificador añadiendo una variable de
negatividad, y esa comparación solo es válida si **el conjunto de prueba es exactamente el
mismo**. Si cada persona volviera a partir por su cuenta, cualquier diferencia de métrica
podría deberse al cambio de partición y no a la variable nueva.

## Ausencia de fuga de información

El vectorizador va **siempre** dentro de un `Pipeline` de scikit-learn, nunca ajustado por
separado sobre el conjunto completo. Cuando se ejecuta `pipe.fit(X_train, y_train)`, el
vocabulario y los pesos IDF se calculan usando únicamente el conjunto de entrenamiento. Las
palabras que solo existen en el conjunto de prueba simplemente no forman parte del
vocabulario, que es exactamente lo que ocurriría con un tweet nuevo en producción. La misma
regla se aplica al `OneHotEncoder` del experimento con `keyword`.

## Baseline

Antes de entrenar nada hay que fijar el piso. `DummyClassifier(strategy='most_frequent')`
responde siempre "no desastre" sin mirar el texto. Alcanza una **exactitud del 57.38%**, que
es exactamente la proporción de la clase mayoritaria.

Ese número es la razón por la que la exactitud no puede ser la métrica de selección de este
laboratorio. Un modelo que reportara 70% de exactitud suena razonable hasta que se recuerda
que no leer el texto en absoluto ya alcanza un 57%. Su macro-F1, en cambio, es **0.3646**: la
media entre un F1 de 0.7292 en la clase que siempre acierta y un F1 de 0 en la clase que
nunca detecta. La macro-F1 sí captura que el modelo es inútil para lo único que importa.

## Combinaciones evaluadas

Se evaluaron cinco combinaciones además del baseline, todas con `min_df=2` —que descarta los
términos presentes en un solo documento, necesario porque el análisis de n-gramas advirtió
que el vocabulario de bigramas es muy disperso— y con `class_weight='balanced'` en los
algoritmos que lo admiten, para compensar el desbalance 57/43 penalizando más los errores
sobre la clase de interés. La **Figura 1** compara los resultados y la tabla completa está en
`results/tables/metricas_preliminares.csv`.

| Modelo | Exactitud | F1 desastre | macro-F1 |
|---|---|---|---|
| **TF-IDF (1,2) + Regresión logística** | **0.7923** | **0.7482** | **0.7857** |
| TF-IDF (1,1) + Regresión logística | 0.7923 | 0.7478 | 0.7856 |
| TF-IDF (1,2) + LogReg sobre `text_stem` | 0.7896 | 0.7437 | 0.7826 |
| TF-IDF (1,2) + Naive Bayes multinomial | 0.7902 | 0.7125 | 0.7737 |
| Bolsa de palabras (1,2) + LinearSVC | 0.7602 | 0.7036 | 0.7511 |
| Baseline (clase mayoritaria) | 0.5738 | 0.0000 | 0.3646 |

Las cuatro combinaciones superan al baseline por más de 0.38 puntos de macro-F1, así que el
texto contiene señal aprovechable y el modelo aprende algo real. Tres lecturas concretas:

**Los bigramas casi no aportan.** El rango (1,2) obtiene 0.7857 y el (1,1) obtiene 0.7856:
una diezmilésima de diferencia, muy por debajo de lo atribuible a la elección de partición. La
sección de n-gramas había recomendado incluirlos porque capturan colocaciones reales como
`suicide bomber` u `oil spill`, y esas colocaciones existen. Lo que muestra este resultado es
que **su aporte de información ya estaba contenido en los unigramas correspondientes** para
efectos de clasificación. Se conserva el rango (1,2) porque no perjudica, pero queda
registrado que hoy no es la palanca que mejora el modelo.

**Naive Bayes cambia el equilibrio, no el nivel.** Su exactitud es casi idéntica a la de la
regresión logística (0.7902 contra 0.7923), pero reparte los errores de forma muy distinta:
logra la mayor precisión de todos en la clase desastre (0.8568) a costa de la peor
exhaustividad (0.6097). Cuando dice que un tweet es un desastre acierta más que ningún otro,
pero deja pasar casi cuatro de cada diez desastres reales. Para un problema donde no detectar
un desastre real es el error caro, ese intercambio es el equivocado. Es precisamente lo que la
macro-F1 penaliza y la exactitud habría ocultado.

**El stemming no supera a la lematización**, lo que respalda la elección de `text_clean` que
la sección de preprocesamiento ya había hecho por motivos de interpretabilidad.

## Análisis de los errores

La **Figura 2** muestra la matriz de confusión del mejor modelo. Comete 311 errores sobre
1,497 tweets de prueba, y no están repartidos por igual:

| | Predicho no desastre | Predicho desastre |
|---|---|---|
| **Real no desastre** | 724 | 135 |
| **Real desastre** | 176 | 462 |

**176 desastres reales no fueron detectados**, frente a 135 falsas alarmas. Es el error más
costoso para este problema —`config.py` declara `CLASE_POSITIVA = 1` con el comentario de que
no detectar un desastre real es el error caro— y ocurre pese a haber usado
`class_weight='balanced'`.

La asimetría tiene la explicación que el análisis exploratorio anticipó. La clase "no
desastre" se define por **ausencia**: incluye cualquier tema que no sea una catástrofe y su
vocabulario es difuso. La clase "desastre" tiene un campo semántico cerrado. Un modelo lineal
aprende bien ese campo cerrado, pero cuando un tweet de desastre real está redactado sin el
vocabulario típico, no tiene de dónde agarrarse y lo manda al cajón de "todo lo demás".

## Qué aprendió el modelo

La regresión logística es un modelo lineal, así que cada término tiene un coeficiente cuyo
signo indica hacia qué clase empuja. Eso permite auditar lo aprendido en lugar de aceptarlo
como una caja negra. La **Figura 3** muestra los 15 términos más influyentes de cada lado,
sobre un vocabulario de 9,308 términos, y la tabla está en
`results/tables/modelo_coeficientes_top15.csv`.

Del lado de desastre aparecen `fire`, `hiroshima`, `storm`, `california`, `wildfire`, `flood`,
`bombing`, `earthquake` y `killed`. Aparece también `urlweb`, el token que sustituyó a las
URLs, con un coeficiente alto: **el modelo confirma por su cuenta la decisión de no borrar los
enlaces**, tomada a partir de la diferencia 67.35% contra 41.73% medida en el análisis
exploratorio. Si se hubieran eliminado, esa señal se habría perdido.

El lado contrario es más revelador. Junto al vocabulario conversacional esperable —`love`,
`new`, `let`, `get`, `want`, `song`, `lol`— aparecen tres términos de catástrofe pura:
**`explode`, `avalanche` y `traumatised` empujan hacia NO desastre.** No es un error: es la
medición directa del uso figurado. Se tuitea *"my head is going to explode"* o *"an avalanche
of emails"* mucho más de lo que se reporta una explosión o un alud real.

Esto refina el hallazgo central del laboratorio. No es solo que el vocabulario de desastre sea
ambiguo: **algunos términos de desastre son activamente contraindicativos**, con un efecto
opuesto al que sugeriría la intuición.

### Validación cruzada con el análisis de n-gramas

La sección de n-gramas calculó, sin entrenar ningún modelo, el log-ratio
`log( P(t | desastre) / P(t | no_desastre) )` de cada término. Comparar ambos resultados es
una validación genuina porque los métodos son independientes: uno optimiza una función de
pérdida, el otro solo cuenta frecuencias.

La **correlación de Spearman entre el coeficiente aprendido y el log-ratio contado es 0.8590**
sobre los 2,549 términos presentes en ambos análisis. Es alta: el modelo ordena el vocabulario
prácticamente igual que el conteo.

El solapamiento entre los extremos es más modesto —4 de 15 términos hacia desastre y 5 de 15
hacia no desastre están entre los 50 más extremos del log-ratio— y esa diferencia también
informa. El log-ratio evalúa cada palabra **aislada**; la regresión logística asigna
coeficientes **con las demás palabras presentes**, repartiendo el peso entre términos
correlacionados. Que el orden global coincida tanto y los extremos difieran algo es lo
esperable de dos métodos que miden lo mismo por caminos distintos.

Los términos que coinciden en ambos extremos son los más significativos: `bag` es el que más
empuja hacia "no desastre" en los dos análisis, y viene casi por completo de `body bag`, que
el análisis exploratorio ya había señalado como el caso más claro de vocabulario de catástrofe
en sentido no literal. `ruin`, `wrecked` y `blew` están ahí por lo mismo. **El modelo aprendió
por su cuenta el hallazgo central del análisis exploratorio**, sin que nadie se lo programara.

## Hipótesis 1: ¿ayuda `keyword` como característica? No

El análisis exploratorio concluyó que `keyword` *"debe incorporarse como característica al
modelo del ejercicio 6"*, con un argumento sólido: catálogo cerrado de 221 términos, 0.80% de
nulos y tasas de desastre del 0% al 100%. La recomendación se comprobó antes de darla por
buena, añadiéndola con `OneHotEncoder` dentro de un `ColumnTransformer`.

| Modelo | Exactitud | F1 desastre | macro-F1 |
|---|---|---|---|
| TF-IDF (1,2) + LogReg | 0.7923 | 0.7482 | **0.7857** |
| TF-IDF (1,2) + `keyword` one-hot + LogReg | 0.7729 | 0.7297 | 0.7669 |

**Añadir `keyword` empeora el modelo en 0.0187 puntos de macro-F1.** La explicación se midió:
**la keyword ya aparece escrita dentro del texto del tweet en el 94.10% de los casos**, así
que el TF-IDF ya la había capturado como un token más. El one-hot no aporta información
nueva; aporta 222 columnas dispersas, la mayoría con muy pocos ejemplos, que diluyen los pesos
y favorecen el sobreajuste.

Conviene separar dos cosas que es fácil confundir. Que `keyword` sea **informativa** es cierto
y el análisis exploratorio lo demostró. Que sea **informativa de forma adicional al texto** es
una afirmación distinta, y es la que resulta falsa. Una variable redundante con lo que el
modelo ya tiene no suma, y si llega en forma dispersa, resta.

Esto no invalida el análisis exploratorio: `keyword` sigue siendo la herramienta de
diagnóstico más útil del laboratorio, y es lo que permite el análisis de la sección siguiente.
Lo que cambia es su papel: **sirve para entender el problema, no para alimentar el modelo.**

## Hipótesis 2: ¿se concentran los errores en el lenguaje figurado? Sí

El análisis exploratorio predijo que los errores *"deberían concentrarse en las keywords de
tasa intermedia"*. Para comprobarlo se agrupan las keywords por su tasa de desastre y se mide
la tasa de error en cada grupo. **La tasa de cada keyword se calcula únicamente sobre el
conjunto de entrenamiento**: usar el de prueba para construir el criterio con el que después
se juzgan los errores sobre ese mismo conjunto sería fuga de información. La **Figura 4**
presenta el resultado y la tabla está en `results/tables/modelo_errores_por_keyword.csv`.

| Grupo de keyword | Tweets | Tasa de error |
|---|---|---|
| Extrema baja (tasa ≤ 0.2) | 468 | 16.88% |
| **Intermedia (0.2 < tasa ≤ 0.8)** | **855** | **25.03%** |
| Extrema alta (tasa > 0.8) | 174 | 10.34% |

**La hipótesis se confirma.** Un tweet cuya keyword es ambigua tiene casi dos veces y media
más probabilidad de ser mal clasificado que uno cuya keyword es inequívoca.

El detalle por keyword lo hace más concreto: `natural disaster` falla en 6 de 8 tweets y `war
zone` en 6 de 9. Son expresiones que la gente usa constantemente como hipérbole —*"my kitchen
is a war zone"*— y el modelo, que solo ve palabras, no puede distinguir la hipérbole del
reporte literal. `lightning` falla en la mitad de sus casos por lo mismo.

Esto responde con evidencia a la pregunta del enunciado sobre **cómo se planea abordar el
contexto**. El problema no es que falten palabras: es que las mismas palabras significan cosas
distintas según el contexto, y una bolsa de palabras no representa contexto por construcción.
Los bigramas son el primer intento de capturarlo y ya se midió que aportan poco. La conclusión
honesta es que **superar este techo exige una representación que distinga sentido literal de
figurado**, y eso son embeddings contextuales, no n-gramas.

## Robustez de la elección

Toda la comparación anterior descansa sobre una única partición. Para comprobar que la ventaja
no es un accidente de esa partición, se repitió la evaluación con validación cruzada
estratificada de 5 pliegues **sobre el conjunto de entrenamiento**; el conjunto de prueba no
participa. La tabla está en `results/tables/modelo_validacion_cruzada.csv`.

| Modelo | macro-F1 medio | Desviación | Mínimo | Máximo |
|---|---|---|---|---|
| TF-IDF (1,2) + LogReg | **0.7954** | 0.0130 | 0.7769 | 0.8122 |
| TF-IDF (1,1) + LogReg | 0.7946 | 0.0142 | 0.7717 | 0.8088 |
| TF-IDF (1,2) + MultinomialNB | 0.7889 | 0.0121 | 0.7655 | 0.7987 |
| TF-IDF (1,2) + `keyword` + LogReg | 0.7842 | 0.0080 | 0.7720 | 0.7950 |

La validación cruzada confirma el orden del conjunto de prueba y despeja las dudas abiertas.
La caída al añadir `keyword` **se reproduce**, lo que descarta que fuera un accidente de una
partición desafortunada y confirma el diagnóstico de redundancia. Sobre los bigramas, la
diferencia entre (1,1) y (1,2) sigue siendo mucho menor que la variabilidad entre pliegues
(±0.0130), lo que refuerza que no hay evidencia de que aporten.

## Modelo preliminar seleccionado

**TF-IDF con rango de n-gramas (1,2) y `min_df=2`, más regresión logística con
`class_weight='balanced'`, entrenada sobre `text_clean`.**

| Aspecto | Elección | Motivo |
|---|---|---|
| Representación | `TfidfVectorizer(ngram_range=(1,2), min_df=2)` | TF-IDF pondera a la baja los términos presentes en muchos documentos, que es el problema identificado con las palabras compartidas entre clases |
| Algoritmo | `LogisticRegression(class_weight='balanced', max_iter=1000)` | Mejor macro-F1 y coeficientes auditables término a término |
| Texto | `text_clean` (lematizado, con números) | Supera a `text_stem` y conserva palabras reales |
| Partición | 80/20 estratificada, `random_state=42` | Congelada en `data/processed/particion.parquet` |
| Métrica | macro-F1 | El desbalance 57/43 vuelve engañosa la exactitud |

Resultado sobre el conjunto de prueba: **macro-F1 = 0.7857**, exactitud 79.23%, frente a un
baseline de 0.3646. La clase desastre alcanza precisión 0.7739 y exhaustividad 0.7241. El
modelo entrenado queda serializado en `results/modelos/modelo_preliminar.joblib`.

Conviene poner ese 79% en perspectiva. El conjunto contiene 18 textos idénticos etiquetados de
forma contradictoria por anotadores humanos, lo que significa que **ni las personas coinciden
consigo mismas** en los casos límite. El techo alcanzable no es 100%, y buena parte de la
distancia restante corresponde a tweets genuinamente ambiguos.

## Plan para la entrega final

Ordenado por lo que la evidencia de hoy sugiere que rendirá más:

1. **Mover el umbral de decisión.** Los 176 falsos negativos contra 135 falsos positivos
   muestran que el umbral por defecto de 0.5 no es el adecuado para un problema donde no
   detectar un desastre es el error caro. Se elegirá maximizando macro-F1 sobre validación
   cruzada, nunca sobre el conjunto de prueba.
2. **Ampliar a las doce combinaciones** previstas, incorporando `RandomForest` y `LinearSVC`
   con TF-IDF, y ajustando `min_df`, `max_features` y la regularización `C` con `GridSearchCV`
   sobre el conjunto de entrenamiento.
3. **Integrar la variable de negatividad** del ejercicio 10 sobre esta misma partición
   congelada. La comparación será limpia porque el conjunto de prueba no cambia. La
   expectativa, a partir del análisis exploratorio, es que aporte poco: los tweets de desastre
   resultaron más factuales y menos emocionales, no más negativos.
4. **Construir la función de clasificación del ejercicio 7** sobre este pipeline. Recibirá
   texto crudo, le aplicará `pipeline_texto` y devolverá la etiqueta; que el preprocesamiento
   viva dentro del `Pipeline` es lo que garantiza que un tweet nuevo reciba exactamente el
   mismo tratamiento que los de entrenamiento.
5. **No se insistirá con `keyword` como característica** ni con los trigramas. Ambos quedaron
   medidos y ninguno aporta.
