# Descripción de los datos

## Fuente y obtención

El conjunto de datos proviene de la competencia de Kaggle *Natural Language Processing with
Disaster Tweets*. Los tweets fueron recolectados originalmente por la empresa Figure Eight y
distribuidos en su colección *Data For Everyone*; cada uno fue **etiquetado manualmente por
anotadores humanos** según se refiriera o no a un desastre real.

La descarga está automatizada en `src/download_dataset.py`. El script intenta la vía oficial
de Kaggle y, además de descargar, **verifica la huella SHA256 del archivo** y valida su
estructura. Esa verificación no es un adorno: garantiza que las tres personas del equipo
trabajaron sobre exactamente el mismo insumo y permite que cualquiera reproduzca el análisis
comprobando que su copia coincide. La huella del archivo utilizado es
`61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df`.

## Precisión sobre el tamaño del conjunto

El enunciado del laboratorio describe un conjunto de *"más de 10 500 filas y 5 columnas"*.
Conviene precisar esa cifra, porque no corresponde al archivo que se analiza:

| Archivo | Filas | ¿Incluye `target`? |
|---|---|---|
| `train.csv` | 7,613 | Sí |
| `test.csv` | 3,263 | **No** |
| Suma | 10,876 | — |

La cifra del enunciado corresponde a la suma de ambos archivos. El `test.csv` de la
competencia es el conjunto de evaluación ciega: **no contiene la columna `target`**, porque
las etiquetas las conserva Kaggle para puntuar los envíos de los participantes. En
consecuencia no puede usarse para medir el desempeño de ningún modelo de forma local.

Esto tiene una implicación metodológica que condiciona todo el laboratorio: **el análisis
completo se realiza sobre `train.csv`, y la partición entrenamiento/prueba se obtiene
dividiendo ese archivo**. No es una preferencia del equipo sino una restricción impuesta por
la estructura de los datos disponibles.

## Variables originales

El archivo tiene cinco columnas:

| Variable | Tipo | Descripción | Nulos |
|---|---|---|---|
| `id` | entero | Identificador único del tweet. Sin significado semántico. | 0 |
| `keyword` | texto | Palabra clave asociada al tweet, tomada de un catálogo cerrado de 221 términos. Los espacios vienen codificados como `%20`. | 61 (0.80%) |
| `location` | texto | Ubicación de origen, autodeclarada por el usuario y sin normalizar. | 2,533 (33.27%) |
| `text` | texto | Contenido del tweet, incluyendo URLs, menciones, hashtags y entidades HTML sin decodificar. | 0 |
| `target` | entero | Variable respuesta: `1` = desastre real, `0` = no. | 0 |

Los conteos de nulos corresponden al archivo crudo, antes de deduplicar.

A estas se añaden dos variables derivadas, calculadas sobre el texto **crudo** a propósito,
porque describen el tweet tal como se escribió y no tras limpiarlo: `n_palabras` y
`n_caracteres`.

`id` merece una advertencia explícita: es un identificador administrativo y **no debe usarse
como predictor**. Cualquier capacidad predictiva que mostrara sería un artefacto del orden
en que se construyó el archivo, no una relación real.

## Carga y control de calidad

La carga no se realiza con una lectura directa del CSV en cada cuaderno, sino a través de
`src/carga.py`, que centraliza tres responsabilidades: leer, **validar** y **deduplicar**.
Que los tres cuadernos del laboratorio usen la misma función es lo que hace que los
resultados de las tres personas del equipo sean comparables entre sí.

La validación falla de forma temprana y con un mensaje explícito si el archivo no cumple lo
esperado. En particular detecta el error más probable —haber descargado `test.csv` en lugar
de `train.csv`— y lo explica en lugar de producir un fallo confuso más adelante.

### Deduplicación

El archivo crudo contiene textos repetidos. Son dos problemas distintos y se tratan de forma
distinta:

| Situación | Textos | Filas | Tratamiento |
|---|---|---|---|
| Etiquetas contradictorias | 18 | 55 | Se eliminan todas sus filas |
| Duplicados con etiqueta consistente | 51 | 73 | Se conserva la primera aparición |

**De 7,613 filas se eliminan 128 y quedan 7,485.**

Los motivos son diferentes en cada caso. Los **conflictos de etiqueta** son textos idénticos,
palabra por palabra, que aparecen marcados a la vez como desastre y como no desastre. No hay
criterio disponible para decidir cuál anotación es correcta, así que conservarlos solo
introduciría señal contradictoria durante el entrenamiento. Su existencia es en sí misma un
hallazgo relevante: **ni siquiera los anotadores humanos coincidieron entre ellos**, lo que
establece un techo al desempeño que cualquier modelo puede aspirar a alcanzar sobre este
conjunto.

Los **duplicados exactos** se eliminan por una razón metodológica: la deduplicación ocurre
**antes** de partir en entrenamiento y prueba. Si se hiciera en el orden inverso, el mismo
tweet podría quedar a ambos lados de la partición y el modelo sería evaluado sobre textos
que ya vio durante el entrenamiento, produciendo una métrica optimista por una razón
puramente artificial.

## Distribución de la variable respuesta

Tras la deduplicación, las clases quedan repartidas así:

| `target` | Significado | Frecuencia | Porcentaje |
|---|---|---|---|
| 0 | No desastre | 4,297 | 57.41% |
| 1 | Desastre real | 3,188 | 42.59% |

El desbalance es moderado. No alcanza el caso extremo que haría inservible la exactitud,
pero sí es suficiente para volverla engañosa: **un clasificador que respondiera siempre "no
desastre" alcanzaría un 57.41% de exactitud sin haber aprendido absolutamente nada**. Ese es
el piso contra el que debe compararse cualquier modelo del ejercicio 6, y la razón por la
que el laboratorio adopta macro-F1 como métrica de selección, acompañada de precisión y
exhaustividad por clase y de la matriz de confusión.

## Limitaciones del conjunto de datos

- **Las etiquetas son juicios humanos sobre casos genuinamente ambiguos.** Los 18 textos con
  anotación contradictoria son la prueba directa.
- **El uso figurado del vocabulario de catástrofe es frecuente.** Expresiones como *"this
  exam was a disaster"* emplean ese vocabulario sin referirse a un evento real. Es la
  principal fuente de error esperada, y el análisis exploratorio la documenta en detalle.
- **`location` no es un dato fiable.** Es autodeclarado, no verificado y no normalizado.
- **El corpus es en inglés y es histórico.** Las listas de palabras vacías, los lematizadores
  y los léxicos de sentimiento son específicos del idioma, y los resultados no se extrapolan
  a tweets en español ni a la redacción actual de la plataforma.
- **Los tweets fueron escritos por personas.** Aunque el conjunto no incluye nombres de
  usuario, los textos pueden contener información personal y no deben publicarse fuera del
  contexto educativo del curso.
