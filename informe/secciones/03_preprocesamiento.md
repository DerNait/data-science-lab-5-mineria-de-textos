# Limpieza y preprocesamiento

## Por qué hay dos limpiezas y no una

El laboratorio plantea dos tareas de naturaleza distinta sobre el mismo texto. El ejercicio 6
pregunta si un tweet describe un desastre real: es una tarea de tema, y una tarea de tema se
resuelve mejor con un vocabulario reducido a lo temáticamente relevante. Los ejercicios 8 a
10 de la entrega final piden medir la opinión y la negatividad del texto: es una tarea de
sentimiento, y una tarea de sentimiento necesita justo lo que la de tema descarta, sobre todo
mayúsculas, signos de puntuación expresivos y negaciones intactas.

Aplicar una sola limpieza a las dos tareas rompería una de las dos. Por eso `src/preprocesamiento.py`
implementa dos funciones independientes:

- `limpiar_clasificacion`: agresiva, usada hoy para poblar `text_clean` y `text_stem`.
- `limpiar_sentimiento`: conservadora, queda lista y probada, pero se explota en la entrega
  final.

Ambas funciones, junto con `tokenizar` y `pipeline_texto`, se documentan y demuestran paso a
paso en `notebooks/02_preprocesamiento.ipynb`.

## Los pasos de la limpieza agresiva, con ejemplos reales

`limpiar_clasificacion` aplica, en orden, minúsculas, corrección de mojibake y entidades
HTML, sustitución de URLs por un token uniforme, eliminación de menciones, eliminación del
símbolo de hashtag conservando la palabra, eliminación de emoticones, tokenización, y
eliminación de stopwords. La tabla siguiente muestra el efecto de cada paso sobre tweets
reales del corpus, guardada en `results/tables/preproc_ejemplos_pasos.csv`.

| Paso | Antes | Después |
|---|---|---|
| Mojibake y entidades HTML | `Barbados #Bridgetown JAMAICA [control]... Two cars set ablaze` | `Barbados #Bridgetown JAMAICA Two cars set ablaze` |
| Entidad HTML | `Rene Ablaze &amp; Jacinta` | `Rene Ablaze & Jacinta` |
| URL a token uniforme | `@bbcmtd Wholesale Markets ablaze http://t.co/lHYXEOHY6C` | `@bbcmtd Wholesale Markets ablaze urlweb` |
| Mención | `@bbcmtd Wholesale Markets ablaze` | `Wholesale Markets ablaze` |
| Hashtag | `Our Deeds are the Reason of this #earthquake` | `Our Deeds are the Reason of this earthquake` |
| Emoticón | `Ablaze for you Lord :D` | `Ablaze for you Lord` |

Dos decisiones merecen justificación explícita porque van en contra de lo primero que uno
haría con este tipo de texto.

**Las URLs no se borran, se reemplazan por el token `urlweb`.** El análisis exploratorio
inicial midió que la sola presencia de una URL es el patrón más discriminativo de todo el
corpus: aparece en el 67.35% de los tweets de desastre real, contra el 41.73% del resto. Lo
que es ruido es el texto exacto de cada URL, distinto en cada tweet y sin valor de
generalización, no el hecho de que exista un enlace. Borrar la URL entera tiraría esa señal;
reemplazarla por un token fijo la conserva.

**Los números no se eliminan.** `pipeline_texto` acepta un parámetro `quitar_numeros`, y su
valor por defecto es `True`, pero al construir `text_clean` y `text_stem` se llama
explícitamente con `quitar_numeros=False`. El análisis exploratorio midió que los dígitos
también son señal discriminativa: 72.74% de los tweets de desastre real los contienen, contra
49.90% del resto. Sobre `911` en particular, que el enunciado señala de forma explícita, la
verificación confirma que aparece en solo 4 tweets de los 7,485: es un caso anecdótico, y la
decisión correcta es conservar los números en general, no ese dígito en particular.

**El mojibake se corrige, no se decodifica.** El corpus no contiene ningún carácter Unicode
legítimo: todo lo que aparece fuera del rango ASCII es texto UTF-8 mal decodificado al
construir el archivo original. Un intento de recuperar el contenido original mediante una
recodificación falla, porque el archivo ya perdió esa información antes de llegar al
laboratorio. La única opción consistente es eliminar esos caracteres.

## Tokenización y negaciones

La tokenización usa `nltk.word_tokenize`, el tokenizador de Treebank, en vez de una
separación ingenua por espacios. La razón es concreta: ese tokenizador separa las
contracciones en dos piezas. `don't` se convierte en `do` más `n't`. Esa segunda pieza,
`n't`, es exactamente la forma en que el conjunto `NEGACIONES` del módulo espera encontrar la
partícula de negación:

```
NEGACIONES = {"no", "not", "never", "nothing", "neither", "nor", "n't", "without", "none"}
```

`STOPWORDS_INGLES` se construye restándole ese conjunto a la lista estándar de NLTK en
inglés. Tres de esas negaciones, `no`, `not` y `nor`, sí forman parte de la lista estándar, así
que sin esta resta se perderían durante la limpieza. La tarea de hoy es de tema y en
principio no depende de las negaciones, pero se aplica la misma regla en las dos limpiezas
por consistencia con `limpiar_sentimiento`, donde quitar una negación sí invierte la
polaridad de una frase.

Un efecto colateral de usar un tokenizador lingüístico y no una separación por espacios es
que algunos tokens crudos llegan pegados a signos de puntuación que el tokenizador no separa,
por ejemplo `err:509`. `tokenizar` extrae los fragmentos alfanuméricos internos de esos
tokens en vez de descartarlos completos. Sin ese paso, algunos tweets cortos quedaban
completamente vacíos después de la limpieza.

## Stopwords de dominio

Además de la lista estándar de NLTK, se construyó una lista de stopwords específicas de esta
colección, con un criterio explícito y reproducible: sobre el corpus ya sin stopwords
estándar, una palabra es candidata si aparece en más del 1% de los tweets como frecuencia
documental y si el logaritmo de la razón de probabilidades entre clases, con suavizado de
Laplace, tiene valor absoluto menor a 0.25. Ese logaritmo se calcula así:

```
logratio(palabra) = log( P(palabra | desastre) / P(palabra | no_desastre) )
```

Un valor cercano a 0 significa que la palabra se reparte casi por igual entre las dos clases
y por tanto no aporta capacidad de discriminación. El detalle completo, con la tabla de
candidatas, está en `notebooks/02_preprocesamiento.ipynb` y en
`results/tables/preproc_stopwords_dominio.csv`.

El criterio devuelve una palabra que se descarta pese a cumplirlo numéricamente: `death`,
con logratio 0.220, apenas debajo del umbral. Es vocabulario central de la tarea, no ruido de
dominio, así que se excluye por criterio y no solo por el número. Las palabras que sí entran
a `STOPWORDS_DOMINIO` son:

```
{'video', 'still', 'us', 'man', 'rt', 'first', 'world', 'say', 'could'}
```

Vale la pena notar que este criterio, más estricto que una inspección de frecuencia
compartida sin más, descarta candidatas que a primera vista parecían ruido de dominio en el
análisis exploratorio inicial. `people` y `burning` tienen logratio 0.40 y 0.42
respectivamente: son compartidas por las dos clases, pero no en la misma proporción, así que
sí discriminan y no deben eliminarse. La lección es siempre la misma: frecuencia alta e
informatividad son cosas distintas.

## Stemming y lematización

El módulo implementa las dos técnicas de reducción morfológica. Se comparan sobre 10 palabras
reales del corpus en `results/tables/preproc_stem_vs_lema.csv`:

| Original | Stem | Lema |
|---|---|---|
| earthquake | earthquak | earthquake |
| fires | fire | fire |
| running | run | running |
| bodies | bodi | body |
| crashed | crash | crashed |
| evacuated | evacu | evacuated |
| flooding | flood | flooding |
| burning | burn | burning |
| killed | kill | killed |
| disasters | disast | disaster |

El stemming es una operación mecánica: corta sufijos según reglas fijas sin importarle si el
resultado es una palabra real del idioma, por eso `disasters` se convierte en `disast`. La
lematización busca la forma canónica en el diccionario de WordNet y, por defecto, asume que
la palabra es un sustantivo, por eso `running` y `crashed` no cambian: como sustantivos ya son
su propia forma base. `pipeline_texto` usa lematización por defecto para `text_clean` porque
conserva palabras reales y es más interpretable al leer el vocabulario resultante, aunque
`text_stem` se deja disponible para comparar el efecto de ambas técnicas sobre el modelo en la
entrega final.

## Aplicación al conjunto completo

Se generan tres columnas de texto procesado sobre las 7,485 filas del corpus: `text_clean`
con lematización, `text_stem` con stemming y `text_sent` con la limpieza conservadora de
sentimiento. Un solo tweet, `"What's up man?"`, queda sin ningún token en `text_clean`:
`what` y `up` son stopwords estándar del inglés y `man` es una stopword de dominio de esta
colección. Es un tweet de la clase no desastre, y su vacío es coherente con lo que dice: no
tiene ningún contenido temático que reportar. La fila no se elimina, se conserva con
`text_clean` vacío, porque descartarla rompería la alineación con `target` y con las demás
columnas, y un vector de características todo en cero es una representación válida, aunque
poco informativa, de un tweet sin señal temática. Es un único caso sobre 7,485 filas.
