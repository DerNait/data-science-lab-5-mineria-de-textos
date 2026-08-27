# Análisis exploratorio de los datos

El análisis exploratorio de este laboratorio no busca describir el conjunto por descripción,
sino **producir decisiones**: qué conservar durante el preprocesamiento, qué variables
incorporar al modelo y qué errores anticipar. Cada sección termina en una conclusión
accionable para los ejercicios posteriores.

Todo el análisis está en `notebooks/01_carga_y_eda.ipynb`. Las figuras se encuentran en
`results/figures/` y las tablas en `results/tables/`.

## Distribución de la variable respuesta

La **Figura 1** muestra el reparto de las clases: 4,297 tweets que no describen un desastre
real (57.41%) frente a 3,188 que sí (42.59%).

El desbalance es moderado pero suficiente para condicionar la evaluación. Como se argumenta
en la sección de datos, un clasificador trivial que respondiera siempre con la clase
mayoritaria alcanzaría un 57.41% de exactitud. Reportar únicamente exactitud ocultaría por
completo si el modelo detecta o no los desastres reales, que es justamente lo que interesa.
Por eso todos los modelos del ejercicio 6 se comparan mediante **macro-F1**, con la matriz
de confusión y las métricas por clase como respaldo.

## La variable `keyword`: el hallazgo más útil del análisis

`keyword` es una palabra clave asociada a cada tweet, tomada de un catálogo cerrado de 221
términos. La pregunta relevante no es cuántas hay sino **si separan las clases**, lo que se
mide con la tasa de desastre de cada keyword: la proporción de sus tweets etiquetados como
desastre real.

La **Figura 2** presenta las 15 keywords más y menos asociadas a un desastre real, con la
tasa global (0.43) marcada como referencia. Los resultados completos están en
`results/tables/eda_keyword_tasa_desastre.csv`.

Los extremos son casi deterministas. **`derailment`, `debris` y `wreckage` alcanzan una tasa
del 100%**: todos sus tweets describen eventos reales. En el extremo opuesto, **`aftershock`
tiene una tasa del 0%**: ninguno de sus 32 tweets se refiere a una réplica sísmica real.

Ese contraste no es casual y explica el problema central del laboratorio: **el uso figurado
del lenguaje**. Términos como `aftershock`, `ruin`, `blazing`, `screaming` o `blew up`
pertenecen al vocabulario de la catástrofe pero se emplean de forma cotidiana y metafórica.
En cambio `derailment`, `debris` o `wreckage` son términos técnicos que casi nadie usa en
sentido figurado. **La dificultad del problema no está repartida de manera uniforme: se
concentra en el vocabulario que tiene doble vida.**

De aquí salen dos decisiones:

1. **`keyword` debe incorporarse como característica al modelo del ejercicio 6**, además del
   texto. Es información explícita, casi sin nulos y de baja cardinalidad.
2. Los errores del clasificador deberían concentrarse en las keywords de tasa intermedia.
   Es una hipótesis verificable que el análisis de errores del ejercicio 6 debe comprobar.

En el plano técnico, los espacios de las keywords vienen codificados como `%20` en el
archivo original (`oil%20spill`). Se decodifican antes de analizarlas; de lo contrario la
misma keyword se trataría como un token ilegible.

## La variable `location`: por qué se descarta

`location` contrasta con `keyword` de forma instructiva. Ambas son columnas categóricas del
mismo archivo, pero una resulta utilísima y la otra inservible. Lo que las separa no es el
tema sino **el proceso que las generó**: una procede de un catálogo cerrado y la otra de un
campo de texto libre que el usuario escribe en su perfil, sin validación alguna.

La evidencia contra `location`:

- **Un tercio de los tweets no la tiene** (33.03% de nulos).
- **No está normalizada.** `USA`, `United States` y `New York` conviven como valores
  distintos pese a solaparse conceptualmente; lo mismo ocurre con `UK` y `London`.
- **Su cardinalidad es altísima:** 3,321 valores distintos sobre 5,013 tweets con dato, y la
  mayoría aparece una sola vez. Una categoría que aparece una única vez no permite estimar
  nada.
- Al ser texto libre, contiene entradas que no son lugares en absoluto.

**Decisión:** `location` se descarta como predictor. Aprovecharla exigiría geocodificación y
normalización, un esfuerzo que excede el alcance del laboratorio y que difícilmente
compensaría dado el 33% de faltantes. La decisión queda documentada con evidencia en lugar
de omitir la variable sin explicación.

## Longitud de los tweets

La **Figura 3** muestra los histogramas de longitud por categoría, en palabras y en
caracteres. Las dos medidas cuentan historias distintas, y la diferencia entre ambas es lo
interesante.

| Métrica | No desastre | Desastre real | Mann-Whitney |
|---|---|---|---|
| Palabras (media) | 14.67 | 15.14 | p ≈ 2×10⁻⁴ |
| Caracteres (media) | 95.58 | 108.02 | p ≈ 2×10⁻⁴⁵ |

En número de palabras la diferencia es mínima: menos de media palabra. En caracteres es
mucho mayor: unos 12 caracteres. La comparación de los valores p confirma que el contraste
en caracteres es de un orden de magnitud completamente distinto.

**Un tweet de desastre tiene prácticamente el mismo número de palabras, pero esas palabras
son más largas.** Dos explicaciones compatibles: incluyen URLs con mucha mayor frecuencia
—como confirma la sección siguiente— y emplean vocabulario técnico más extenso
(`derailment`, `evacuation`, `catastrophe`) frente al registro coloquial del resto.

Conviene una advertencia sobre la lectura de estos valores p: con 7,485 observaciones casi
cualquier diferencia resulta estadísticamente significativa. Lo relevante no es que el valor
p sea pequeño, sino que **la diferencia en caracteres es lo bastante grande como para ser
aprovechable**, mientras que la de palabras, aun siendo significativa, es demasiado pequeña
para resultar útil por sí sola.

## Patrones de escritura: lo que no conviene borrar

Esta parte del análisis existe para alimentar decisiones del ejercicio 3. El enunciado
propone eliminar URLs, menciones, símbolos, números y emoticones. Antes de borrar conviene
medir **cuánta señal se estaría descartando**, porque lo que es ruido para una tarea puede
ser la señal de otra. La **Figura 4** compara la presencia de cada patrón entre categorías.

| Patrón | No desastre | Desastre real | Diferencia |
|---|---|---|---|
| Contiene URL | 41.73% | 67.35% | **+25.62 pp** |
| Contiene algún dígito | 49.90% | 72.74% | **+22.85 pp** |
| Contiene hashtag | 20.13% | 25.97% | +5.84 pp |
| Contiene MAYÚSCULAS | 18.20% | 21.30% | +3.10 pp |
| Contiene entidad HTML | 5.14% | 4.27% | −0.88 pp |
| Contiene exclamación | 11.99% | 6.02% | **−5.96 pp** |
| Contiene mención (@) | 31.07% | 20.67% | **−10.40 pp** |

**Las URLs son el patrón más discriminativo del conjunto.** Aparecen en el 67.35% de los
tweets de desastre frente al 41.73% de los demás, una diferencia de casi 26 puntos
porcentuales. Tiene sentido: un tweet sobre un evento real suele enlazar la noticia que lo
respalda. Esto obliga a un matiz importante en el preprocesamiento: *el texto* de una URL es
ruido —`http://t.co/k4zoMOF319` no aporta nada y además infla el vocabulario con un token
irrepetible—, pero *el hecho de que exista una URL* es señal valiosa. **La recomendación es
sustituir cada URL por un token uniforme en lugar de eliminarla**, conservando la
información y descartando el ruido.

**Los dígitos siguen el mismo patrón:** 72.74% frente a 49.90%, casi 23 puntos de
diferencia. Los tweets de desastre están llenos de magnitudes, fechas, número de víctimas y
números de carretera. **Eliminar los números a ciegas destruiría señal real.**

Sobre el `911` que el enunciado sugiere considerar de forma específica: aparece en **solo 4
tweets** de los 7,485. Es anecdótico. La verificación resulta útil precisamente porque
corrige la intuición: lo que conviene conservar no es un número concreto sino **los números
en general**, que es donde está la señal.

En dirección contraria, las **menciones** y las **exclamaciones** son más frecuentes en los
tweets que **no** son de desastre. Encajan con un registro conversacional y emocional,
frente al tono informativo y factual de los tweets que reportan eventos reales. Este
hallazgo anticipa un resultado del ejercicio 10: si los tweets de desastre son más factuales
y menos emocionales, una variable de sentimiento podría no distinguirlos tan bien como
parecería a primera vista.

## Emoticones: la pregunta del enunciado tiene respuesta empírica

El enunciado pide revisar si hay emoticones y quitarlos (ejercicio 3), y más adelante
pregunta si valdrá la pena conservarlos para analizarlos (ejercicio 8). La medición responde
ambas preguntas:

| Elemento | Tweets | Porcentaje |
|---|---|---|
| Emojis Unicode | 0 | 0.00% |
| Emoticones ASCII (`:)`, `:D`, `<3`) | 70 | 0.94% |
| Caracteres corruptos (mojibake) | 591 | 7.90% |

**No hay ni un solo emoji Unicode en el corpus.** Los emoticones ASCII aparecen en apenas el
0.94% de los tweets. En cambio, el 7.90% contiene **mojibake**: secuencias como `\x89ÛÏ` que
no son emojis sino texto Unicode mal decodificado al construir el archivo original. Con toda
probabilidad ahí estuvieron los emojis, destruidos antes de que el conjunto llegara a
nosotros.

Las consecuencias son directas. Para el ejercicio 3, no hace falta un tratamiento sofisticado
de emoticones porque prácticamente no existen; lo que sí hace falta es **limpiar el
mojibake**, que afecta a ocho veces más tweets y produce tokens basura. Para el ejercicio 8,
la respuesta a *"¿valdrá la pena dejar los emoticones y analizarlos?"* es **no en este
corpus**, y el motivo es la ausencia de material, no una decisión de diseño.

Los 70 tweets que sí tienen emoticón muestran una tasa de desastre del 17.1%, muy por debajo
del 42.6% global. Es coherente con todo lo anterior —quien escribe `:)` no está reportando
una catástrofe— pero con 70 casos la base es demasiado pequeña para sostener una conclusión
firme, y así se reporta.

## Palabras más frecuentes en cada categoría

La **Figura 5** presenta las 20 palabras más frecuentes de cada categoría y las **Figuras 6,
7 y 8** las nubes de palabras del corpus completo, de los tweets de desastre y de los
demás.

En los tweets de **desastre real** la palabra más repetida es **`fire`**, acompañada de
`news`, `disaster`, `california`, `police`, `suicide`, `killed`, `hiroshima`, `storm`,
`crash`, `emergency` y `nuclear`. Es vocabulario de sucesos y de cobertura periodística.
Aparecen incluso eventos concretos: `california` e `hiroshima` remiten a episodios
específicos presentes durante el periodo de recolección.

En los tweets que **no** son de desastre la palabra más repetida es **`body`**, seguida de
`video`, `love`, `day`, `know`, `back` y `time`: vocabulario conversacional cotidiano, sin
tema común.

Que la palabra más frecuente de la clase negativa sea `body` merece atención. Procede de
keywords como `body bag` y `body bags`, que la Figura 2 ya situaba entre las menos asociadas
a desastre real. Es un ejemplo perfecto del problema central del laboratorio: vocabulario de
catástrofe empleado en sentido no literal.

La comparación de las nubes revela además una **asimetría entre las clases**. La nube de
desastre está dominada por un campo semántico cerrado y reconocible; la de no desastre es
mucho más difusa, porque la clase negativa se define por **ausencia** —no habla de un
desastre real— y eso incluye cualquier cosa. La implicación para el ejercicio 6 es que **la
clase positiva es más aprendible que la negativa**: un modelo puede reconocer el vocabulario
de la catástrofe, pero no puede reconocer "todo lo demás".

Conviene recordar que la nube de palabras es una herramienta de exploración, no de medición:
el tamaño de cada término refleja su frecuencia bruta, sin ponderar por lo informativo que
sea.

## Palabras presentes en ambas categorías

El enunciado pide discutir las palabras con presencia en todas las categorías. Importan por
una razón concreta: **una palabra frecuente en las dos clases no ayuda a distinguirlas**, por
mucho que destaque en una nube.

Siete palabras aparecen en el top 40 de ambas categorías: `burning`, `emergency`, `fire`,
`news`, `people`, `still` y `video`. La tabla
`results/tables/eda_palabras_compartidas.csv` recoge para cada una su frecuencia y
probabilidad en cada clase, junto al logaritmo del cociente entre ambas probabilidades.

No todas las palabras compartidas son iguales, y conviene separarlas en dos grupos:

- **Compartidas y poco informativas**, con log-ratio cercano a cero: `people`, `still` y
  `burning`. Son frecuentes en ambos lados y funcionan como ruido de fondo del dominio. Son
  candidatas naturales a **palabras vacías específicas de esta colección**, además de la
  lista estándar del inglés.
- **Compartidas pero informativas**, con log-ratio apreciable: `fire`, `news` y `emergency`
  se inclinan hacia desastre, mientras que `video` se inclina hacia no desastre. Aparecen en
  ambas clases, pero no en la misma proporción. **Estas no deben eliminarse**: su frecuencia
  bruta engaña, y solo la comparación entre clases revela que sí aportan.

La lección metodológica es que **frecuencia e informatividad son cosas distintas**. Es
exactamente el problema que TF-IDF resuelve al penalizar los términos presentes en muchos
documentos, y la razón por la que el análisis de n-gramas del ejercicio 4 se apoya en
medidas relativas entre clases y no solo en conteos.

## Síntesis: decisiones que produce el análisis exploratorio

**Sobre los datos**

1. Quedan 7,485 tweets tras eliminar 128 filas duplicadas o contradictorias. Los 18 textos
   con anotación contradictoria evidencian que la tarea es ambigua incluso para anotadores
   humanos.
2. El desbalance es moderado (57.41% / 42.59%), por lo que la métrica de selección es
   macro-F1.

**Para el preprocesamiento (ejercicio 3)**

3. Las URLs deben **reemplazarse por un token uniforme**, no borrarse: su presencia es el
   patrón más discriminativo del conjunto.
4. Los números **no deben eliminarse a ciegas**: aparecen en el 72.74% de los tweets de
   desastre frente al 49.90% del resto.
5. Hay que **limpiar el mojibake** (7.90% de los tweets). El tratamiento de emoticones es
   irrelevante en este corpus.
6. Las palabras vacías de dominio deberían incluir `people`, `still` y `burning`, pero **no**
   `fire`, `news`, `emergency` ni `video`.

**Para el modelado (ejercicios 6 y 10)**

7. **`keyword` debe incorporarse como característica**: es la variable estructurada más
   informativa del conjunto.
8. **`location` se descarta**, con la evidencia del 33% de nulos y 3,321 valores sin
   normalizar.
9. Los errores deberían concentrarse en el **uso figurado del vocabulario de catástrofe**.
   Es una hipótesis verificable en el análisis de errores del ejercicio 6.
10. Los tweets de desastre son **más factuales y menos emocionales**, lo que anticipa que la
    variable de sentimiento del ejercicio 10 podría aportar poco. Ese ejercicio deberá
    comprobarlo en lugar de suponerlo.
