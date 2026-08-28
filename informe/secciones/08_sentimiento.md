# Análisis de sentimiento

Esta sección cubre el **ejercicio 8**: determinar las palabras positivas, negativas y neutras,
medir a partir de ellas qué tan positivo o negativo es cada tweet, y responder si vale la pena
conservar los emoticones para analizarlos. Todo el trabajo está en
`notebooks/07_sentimiento.ipynb`, el código reutilizable en `src/sentimiento.py`, y las tablas
y figuras en `results/tables/sentimiento_*.csv` y `results/figures/sentimiento_*.png`.

## La herramienta: por qué VADER

Se usa **VADER** (*Valence Aware Dictionary and sEntiment Reasoner*), del módulo
`nltk.sentiment.vader`. La elección tiene cuatro razones y conviene dejarlas explícitas
porque condicionan todo lo que sigue:

1. **El corpus es en inglés y VADER está calibrado sobre texto de redes sociales**, que es
   exactamente el registro de estos tweets.
2. **Maneja de forma nativa lo que un conteo de palabras no puede**: negación,
   intensificadores, MAYÚSCULAS, signos de exclamación y emoticones. Cada uno de esos
   elementos cambia la polaridad de una frase y todos abundan en Twitter.
3. **Es un enfoque basado en léxico y no requiere datos etiquetados de sentimiento.** Este
   punto no es una comodidad sino una restricción: el conjunto **no tiene etiquetas de
   sentimiento**. Su única etiqueta, `target`, dice si el tweet habla de un desastre real, no
   si es positivo o negativo. Entrenar un clasificador supervisado de sentimiento es
   imposible aquí, así que el enfoque léxico es la única vía disponible.
4. Es el enfoque basado en léxico que describe el material del curso.

Como contraste se usa además **TextBlob**, que el enunciado lista entre los módulos útiles.
Ambas herramientas se referencian al final de esta sección.

## Sobre qué texto se puntúa: el punto fino del ejercicio

La intuición dice que al análisis de sentimiento se le debe dar el texto ya limpio, igual que
al clasificador. **La intuición se equivoca**, y conviene demostrarlo en lugar de afirmarlo.

`text_clean` perdió las mayúsculas, la puntuación, los emoticones y las palabras vacías. Pero
las palabras vacías del inglés incluyen `no` y `not`, y la puntuación incluye el `!`. Es
decir: **`text_clean` eliminó justamente aquello que VADER sabe leer.**

Se corrió VADER sobre cuatro versiones del mismo tweet y se midió cuántos cambian de etiqueta
de polaridad. La **Figura 2** superpone las distribuciones y la tabla está en
`results/tables/sentimiento_versiones_texto.csv`.

**501 tweets, el 6.69% del corpus, cambian de etiqueta** según se puntúe el texto conservador
o el agresivamente limpiado. Y no cambian un poco: los casos de mayor salto **invierten el
signo por completo**.

| Tweet | Conservador | Limpieza agresiva |
|---|---|---|
| *"Another movie theater attack..close to home this time. Thankful for no casualties."* | **+0.8570** | **−0.6395** |
| *"...I'm pretty sure I'm not dead"* | **−0.7218** | **+0.8445** |
| *"Thousands of people were displaced/injured/killed but hey now there's more opportunity..."* | **+0.6632** | **−0.6597** |

El primer caso es el más claro. Con el texto conservador VADER lee `thankful for no
casualties` como lo que es —alivio— y devuelve +0.8570. Tras la limpieza agresiva el texto
queda como `another movie theater attack close home time thankful no casualty...`: al perderse
la estructura de la frase, `attack` y `casualty` pesan como términos sueltos y el alivio se
convierte en catástrofe.

En cambio, la diferencia entre el texto crudo y el conservador es pequeña —solo **114 tweets,
el 1.52%**, cambian de etiqueta— lo que confirma que eliminar URLs y menciones es seguro: no
cargan opinión.

**Conclusión:** el análisis de sentimiento se hace sobre texto conservador. Esto valida la
decisión que el equipo tomó en el ejercicio 3 de mantener **dos limpiezas separadas** en lugar
de una sola, y la valida con una medición y no con un argumento.

## Nivel de palabra: qué palabras son positivas, negativas y neutras

El enunciado pide *"determinar las palabras positivas, negativas o neutras"*. El léxico de
VADER trae más de 7,500 términos puntuados a mano por anotadores humanos en una escala de −4 a
+4, pero listarlos todos no diría nada sobre **este** corpus. Lo que interesa es la
intersección: qué palabras del léxico aparecen efectivamente en estos tweets. La **Figura 1**
las presenta y la tabla completa está en `results/tables/sentimiento_lexico_corpus.csv`.

**Solo 1,323 de las 13,348 palabras del vocabulario están en el léxico**, un 9.91%: 729
negativas y 594 positivas. Esa cifra por sí sola ya es un resultado: **nueve de cada diez
palabras del corpus no cargan sentimiento**. Son nombres propios, lugares, números, verbos
neutros y vocabulario técnico. El sentimiento de un tweet se decide con una fracción pequeña
de lo que dice.

| Más negativas | Valencia | Frecuencia | | Más positivas | Valencia | Frecuencia |
|---|---|---|---|---|---|---|
| `slavery` | −3.8 | 1 | | `greatest` | 3.2 | 8 |
| `rape` | −3.7 | 5 | | `freedom` | 3.2 | 5 |
| `murder` | −3.7 | 43 | | `love` | 3.2 | 107 |
| `kill` | −3.7 | 68 | | `paradise` | 3.2 | 1 |
| `terrorist` | −3.7 | 48 | | `best` | 3.2 | 75 |

La columna de frecuencia es la que hace interesante la tabla, porque **valencia y presencia
son cosas distintas**. `slavery` tiene la valencia más negativa del corpus pero aparece **una
sola vez**: su carga es máxima y su influencia agregada es nula. `kill` (68 apariciones),
`terrorist` (48) y `murder` (43) sí mueven la aguja. Del lado positivo pasa lo mismo:
`paradise` aparece una vez y `love` aparece 107. **Las palabras que determinan el sentimiento
del corpus no son las más extremas sino las que combinan carga alta con frecuencia alta.**

## Nivel de tweet: contar palabras frente a medir polaridad

El enunciado continúa: *"teniendo en cuenta la cantidad de palabras positivas y negativas del
tweet determine qué tan positivo, negativo o neutral es el mismo"*. Se implementó literalmente
con la función `contar_palabras`, que cuenta cuántos tokens de cada tweet son positivos,
negativos o neutros según el léxico y define el **balance** como positivas menos negativas.
Esas cuatro columnas forman parte del conjunto entregado.

Ese conteo es deliberadamente ingenuo, y ahí está su valor: **no maneja negación**. Contrastarlo
con el `compound` de VADER demuestra, con casos del propio corpus, por qué hace falta un modelo
con reglas y no basta con contar.

**El conteo y VADER coinciden en el 85.12% de los tweets.** El 15% restante es donde está la
lección. Hay **116 tweets que el conteo declara negativos y VADER declara positivos**, y **94
de ellos —el 81%— contienen una partícula de negación explícita**:

| Tweet | Conteo | VADER |
|---|---|---|
| *"No way...I can't eat that shit"* | 0 pos / 2 neg → **negativo** | +0.1838 → **positivo** |
| *"People who say it cannot be done should not interrupt those who are doing it."* | 0 pos / 1 neg → **negativo** | +0.2584 → **positivo** |
| *"Bioterror lab faced secret sanctions. RickPerry doesn't make the cut..."* | 0 pos / 1 neg → **negativo** | +0.2057 → **positivo** |

El conteo suma valencias sin mirar qué palabra viene antes; VADER aplica una regla que
invierte y atenúa la polaridad del término que sigue a una negación. **Es exactamente el
problema de la negación que describe el material del curso**, medido aquí sobre el corpus
propio. Que el 81% de las discrepancias en esa dirección se expliquen por una negación
explícita confirma que el mecanismo es ese y no otro.

Por eso las columnas de conteo se conservan —el enunciado las pide— pero **la etiqueta de
polaridad oficial del laboratorio se toma del `compound`**.

## Los emoticones: la pregunta del enunciado, respondida con evidencia

El ejercicio 8 pregunta: *"¿valdrá la pena dejar los emoticones y analizarlos?"*

El análisis exploratorio ya había medido que el corpus **no tiene ni un solo emoji Unicode** y
que los emoticones ASCII aparecen en menos del 1% de los tweets. Con esos números la respuesta
parecería ser "no importa". Pero hay un detalle que cambia el planteamiento y que solo se ve al
mirar el léxico: **VADER ya trae los emoticones puntuados**.

| Emoticón | Valencia en VADER | | Token traducido | Valencia |
|---|---|---|---|---|
| `:)` | **2.0** | | `emoji_feliz` | **ausente** |
| `:D` | **2.3** | | `emoji_triste` | **ausente** |
| `:(` | **−1.9** | | `emoji_amor` | **ausente** |
| `<3` | **1.9** | | | |

La limpieza conservadora del ejercicio 3 **traduce** los emoticones a tokens de palabra: `:)`
se convierte en `emoji_feliz`. Pero esos tokens **no existen en el léxico**, así que VADER los
ignora por completo. La traducción destruye justamente la señal que la herramienta sabía leer.

El efecto está medido: de los **52 tweets** con emoticón ASCII según los patrones de
`preprocesamiento.py`, **19 cambian de etiqueta de polaridad —el 36.54%—** según se les deje el
emoticón literal o traducido. El `compound` medio del grupo cae de **+0.2175 a +0.0611**, es
decir se pierde alrededor de tres cuartas partes de la señal que aportaban.

**La respuesta, en dos partes:**

1. **Sí vale la pena conservarlos y analizarlos, y además conservarlos literales**, sin
   traducir, porque la herramienta ya sabe leerlos. Por eso el análisis se hace sobre una
   columna `text_vader` creada para esto, y no sobre `text_sent`.
2. **En este corpus concreto da casi igual**, porque son 52 tweets sobre 7,485, el 0.69%. El
   efecto agregado sobre cualquier métrica global es despreciable. La conclusión importa como
   criterio de método —traducir un símbolo que la herramienta ya entiende es perder información
   gratis— más que como mejora de resultados aquí.

Conviene precisar que la traducción a `emoji_feliz` **no es un error del ejercicio 3**: es la
decisión correcta para un clasificador de bolsa de palabras, donde el tokenizador descartaría
`:)` como puntuación suelta y la señal se perdería del todo. Lo que muestra este análisis es
que **la limpieza óptima depende de la herramienta que consume el texto, no solo de la tarea**.

Un dato secundario: esos 52 tweets tienen una tasa de desastre real del **13.46%**, muy por
debajo del 42.6% global. Quien escribe `:)` no suele estar reportando una catástrofe. Con 52
casos la base es pequeña, y así se reporta.

## Umbrales de polaridad

Los umbrales son los estándar de VADER y están declarados en `src/config.py`, no incrustados
en el código, porque **son una decisión del equipo y no una ley**:

- `compound >= 0.05` → positivo
- `compound <= -0.05` → negativo
- entre ambos → neutro

Con ellos, el corpus se reparte en **49.25% negativo, 25.42% neutro y 25.33% positivo**. La
tabla `results/tables/sentimiento_umbrales.csv` muestra la sensibilidad:

| Umbral | Negativo | Neutro | Positivo |
|---|---|---|---|
| ±0.05 | 49.25% | 25.42% | 25.33% |
| ±0.10 | 48.30% | 27.55% | 24.15% |
| ±0.20 | 45.91% | 31.92% | 22.18% |
| ±0.30 | 41.26% | 39.39% | 19.36% |

El reparto es **estable**. Al ampliar la banda neutra los tweets de polaridad débil —los
cortos, con pocas palabras cargadas— se reclasifican como neutros, que es la dirección
esperable, pero los negativos siguen siendo el grupo mayoritario en todos los umbrales
probados. Que la conclusión principal no dependa del umbral elegido es lo que permite
reportarla sin advertencias.

## Contraste con TextBlob

Medir sentimiento con una sola herramienta deja sin responder si el resultado es una propiedad
del corpus o del método. Se repitió la medición con TextBlob y la matriz de acuerdo está en
`results/tables/sentimiento_vader_vs_textblob.csv`.

**Las dos herramientas coinciden en apenas el 47.07% de los tweets**, con una correlación de
Spearman de **0.4014**. Es un desacuerdo grande y es, en sí mismo, el hallazgo de esta
subsección.

La causa se lee en la matriz: **TextBlob clasifica como neutro el 49.65% del corpus, casi el
doble que VADER (25.42%)**. La discrepancia mayor está en los 1,777 tweets que VADER llama
negativos y TextBlob llama neutros. La explicación es de diseño: TextBlob usa el léxico de
*pattern*, construido para prosa general —reseñas, texto editado— y **no incluye emoticones,
jerga de redes sociales ni reglas para mayúsculas o signos de exclamación**. Frente a un tweet
se queda sin vocabulario que puntuar y devuelve 0.

La conclusión metodológica es que **la medición de sentimiento es sensible al método**, y
reportar una cifra de una sola herramienta sin contraste transmite más certeza de la que
existe. Para este corpus se elige VADER por estar calibrado al dominio, pero todos los
resultados de esta sección y de la siguiente deben leerse como **"según VADER"**, no como una
propiedad absoluta de los tweets.

## La variable de negatividad

Para el ejercicio 10 se define:

```
negatividad = (1 - compound) / 2
```

Es una transformación lineal del `compound`, que vive en [−1, 1], al rango [0, 1]. Un tweet
máximamente positivo da negatividad 0; uno máximamente negativo da 1; uno neutro da 0.5. Se
define en [0, 1] a propósito: es el mismo rango en el que TF-IDF produce sus valores, así que
puede concatenarse a la matriz de características sin escalar y sin que su magnitud domine
artificialmente sobre la de los términos.

Conviene anotar una propiedad que importa para interpretar el ejercicio 10: al ser una
transformación **monótona** del `compound`, para un modelo lineal aporta exactamente la misma
información que el `compound` mismo. La diferencia es de interpretación, no de contenido.

El conjunto anotado se entrega en `data/processed/tweets_con_sentimiento.parquet` con 7,485
filas y 21 columnas: las cuatro componentes de VADER (`vader_neg`, `vader_neu`, `vader_pos`,
`vader_compound`), la etiqueta de `polaridad`, la `negatividad`, las dos medidas de TextBlob,
las cuatro columnas de conteo del ejercicio 8 y las versiones del texto. Se verificó que sus
`id` coinciden con los de la partición congelada del ejercicio 6, de modo que el
reentrenamiento del ejercicio 10 pueda compararse limpiamente.

## Referencias de esta sección

- **VADER:** Hutto, C.J. & Gilbert, E.E. (2014). *VADER: A Parsimonious Rule-based Model for
  Sentiment Analysis of Social Media Text*. Eighth International Conference on Weblogs and
  Social Media (ICWSM-14). Se usa a través de `nltk.sentiment.vader.SentimentIntensityAnalyzer`.
- **NLTK:** Bird, S., Klein, E. & Loper, E. (2009). *Natural Language Processing with Python*.
  O'Reilly. Módulos usados: `nltk.sentiment.vader`, `nltk.corpus.stopwords`,
  `nltk.stem.WordNetLemmatizer`, `nltk.word_tokenize`.
- **TextBlob:** Loria, S. *TextBlob: Simplified Text Processing*.
  https://textblob.readthedocs.io/ — usado como contraste; su análisis de polaridad se apoya
  en el léxico de *pattern*.
