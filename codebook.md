# Codebook — Natural Language Processing with Disaster Tweets

## Fuente

El conjunto de datos proviene de la competencia de Kaggle
[*Natural Language Processing with Disaster Tweets*](https://www.kaggle.com/competitions/nlp-getting-started/data)
(`nlp-getting-started`). Los tweets fueron recolectados originalmente por la empresa
**Figure Eight** y distribuidos en su colección *Data For Everyone*. Cada tweet fue
etiquetado **manualmente por anotadores humanos** según se refiriera o no a un desastre
real.

El repositorio conserva el archivo crudo sin modificar en `data/raw/train.csv` y genera
cualquier muestra o transformación en `data/processed/`. El CSV no se versiona: se obtiene
ejecutando `python src/download_dataset.py`.

**Huella del archivo utilizado** (SHA256), para que cualquiera pueda verificar que trabaja
sobre exactamente el mismo insumo:

```
61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df
```

## Sobre el tamaño del conjunto de datos

El enunciado del laboratorio indica que el conjunto está formado por *"más de 10 500 filas
y 5 columnas"*. Conviene precisar esa cifra, porque no corresponde al archivo que se
analiza:

| Archivo | Filas | ¿Trae `target`? |
|---|---|---|
| `train.csv` | 7,613 | Sí |
| `test.csv` | 3,263 | **No** |
| Suma de ambos | 10,876 | — |

La cifra del enunciado corresponde a la **suma de los dos archivos**. El `test.csv` de la
competencia es el conjunto de evaluación ciega: **no contiene la columna `target`**, porque
las etiquetas las conserva Kaggle para puntuar los envíos. Por lo tanto no puede usarse
para medir el desempeño de ningún modelo de forma local.

**Consecuencia metodológica:** todo el laboratorio trabaja sobre `train.csv`, y la partición
entrenamiento/prueba se obtiene dividiendo ese archivo. Es una decisión forzada por la
estructura de los datos, no una preferencia.

## Variables originales

| Variable | Tipo | Descripción | Nulos |
|---|---|---|---|
| `id` | entero | Identificador único del tweet dentro del conjunto. No tiene significado semántico y no debe usarse como predictor. | 0 |
| `keyword` | texto | Una palabra clave asociada al tweet, extraída de un catálogo cerrado de 221 términos (`ablaze`, `earthquake`, `wreckage`...). Puede estar en blanco. Los espacios vienen codificados como `%20`. | 61 (0.80%) |
| `location` | texto | Ubicación desde la que se envió el tweet. Es **texto libre autodeclarado por el usuario**, sin normalizar ni verificar: contiene países, ciudades, coordenadas, bromas y cadenas que no son lugares. | 2,533 (33.27%) |
| `text` | texto | Contenido del tweet. Incluye URLs, menciones, hashtags, emojis y entidades HTML sin decodificar (`&amp;`, `&gt;`, `&lt;`). | 0 |
| `target` | entero | Variable respuesta. `1` = el tweet se refiere a un desastre real; `0` = no. | 0 |

Los conteos de nulos corresponden al archivo crudo, antes de deduplicar.

## Variables derivadas

Las construye `src/carga.py` sobre el texto **crudo**, a propósito: describen el tweet tal
como se escribió, no tras limpiarlo.

| Variable | Tipo | Descripción |
|---|---|---|
| `n_palabras` | entero | Número de palabras del tweet crudo, separando por espacios. |
| `n_caracteres` | entero | Longitud en caracteres del tweet crudo. |

Las variables de texto procesado (`text_clean`, `text_stem`, `text_sent`) las produce
`src/preprocesamiento.py` y están documentadas en la sección de preprocesamiento del
informe.

## Regla de deduplicación

El archivo crudo contiene textos repetidos, y algunos repetidos **con etiquetas
contradictorias**. Son dos problemas distintos y `src/carga.py` los trata por separado.

| Situación | Filas afectadas | Tratamiento |
|---|---|---|
| Textos con etiqueta contradictoria (18 textos distintos aparecen marcados a la vez como desastre y como no desastre) | 55 | Se eliminan **todas** sus filas |
| Textos duplicados con etiqueta consistente | 73 | Se conserva la primera aparición |

**Resultado:** de 7,613 filas se eliminan 128 y quedan **7,485**.

Las dos decisiones tienen motivos distintos:

- Los **conflictos de etiqueta** son ruido de anotación. No hay criterio para decidir cuál
  de las dos etiquetas es correcta, así que conservarlos solo introduce señal contradictoria
  en el entrenamiento. Se eliminan y se deja constancia de cuántos eran, porque su
  existencia es en sí misma un hallazgo sobre la calidad del conjunto.
- Los **duplicados exactos** se eliminan **antes** de partir en entrenamiento y prueba. Si
  no se hiciera, el mismo tweet podría quedar en ambos lados de la partición y la métrica
  saldría optimista por una razón puramente artificial.

## Distribución de la variable respuesta

Tras deduplicar:

| `target` | Significado | Frecuencia | Porcentaje |
|---|---|---|---|
| 0 | No desastre | 4,297 | 57.41% |
| 1 | Desastre real | 3,188 | 42.59% |

El desbalance es moderado. No es el caso extremo de 97/3 que haría inútil la exactitud,
pero sí es suficiente para que un clasificador trivial que responda siempre "no desastre"
alcance un 57.41% de exactitud sin haber aprendido nada. Por eso la métrica de selección
del laboratorio es **macro-F1**, acompañada de precisión y exhaustividad por clase y de la
matriz de confusión.

## Limitaciones

- **Las etiquetas son juicios humanos sobre casos genuinamente ambiguos.** La existencia de
  18 textos con etiquetas contradictorias lo demuestra: ni siquiera los anotadores
  coincidieron consigo mismos. El techo de desempeño alcanzable está limitado por esa
  ambigüedad, no solo por el modelo.
- **El uso figurado del lenguaje de desastres es frecuente y difícil.** Expresiones como
  *"this exam was a disaster"* o *"my life is on fire"* usan vocabulario de catástrofe sin
  referirse a un evento real. Es la principal fuente de error esperada.
- **`location` no es un dato fiable.** Es autodeclarado, no verificado y no normalizado. Su
  utilidad como predictor debe demostrarse, no suponerse.
- **El corpus es en inglés y es histórico.** Las herramientas de un idioma no se trasladan a
  otro: las listas de palabras vacías, los lematizadores y los léxicos de sentimiento son
  específicos del inglés. Los resultados no se extrapolan a tweets en español ni a la
  redacción actual de la plataforma.
- **`id` no debe usarse como predictor.** Es un identificador administrativo; cualquier
  poder predictivo que muestre sería un artefacto del orden del archivo.
- **Los tweets son producidos por personas identificables.** Aunque el conjunto no incluye
  nombres de usuario, los textos pueden contener información personal. No deben publicarse
  fuera del contexto educativo del curso.

## Referencias

- Kaggle. *Natural Language Processing with Disaster Tweets*.
  https://www.kaggle.com/competitions/nlp-getting-started/data
- Figure Eight. *Data For Everyone* (colección original de la que procede el conjunto).
