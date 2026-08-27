# N-gramas, frecuencias y probabilidades

Todo el análisis de esta sección se construye sobre `text_clean`, la versión lematizada y
sin stopwords descrita en la sección anterior, y se detalla en
`notebooks/03_ngramas.ipynb`. Los números se conservan en `text_clean`, así que siguen
disponibles para este análisis.

## Unigramas, bigramas y trigramas por clase

Se cuentan unigramas, bigramas y trigramas por separado, siempre por clase. La probabilidad
de cada n-grama dentro de su clase es su frecuencia entre el total de n-gramas de esa clase.
Las tablas completas están en `results/tables/ngramas_unigramas_top20.csv`,
`ngramas_bigramas_top20.csv` y `ngramas_trigramas_top20.csv`, y la Figura 1 muestra los
unigramas más frecuentes de cada clase.

`urlweb` domina la frecuencia de unigramas en las dos clases: reemplaza cualquier URL, y las
URLs son comunes en ambas. Aun así aparece con más frecuencia relativa en desastre real,
coherente con la diferencia de 67.35% contra 41.73% que ya había medido el análisis
exploratorio inicial. El resto de la lista es la que aporta información: `fire`, `news`,
`disaster`, `california`, `suicide` y `police` dominan el lado de desastre real, vocabulario
temático y de cobertura periodística; `like`, `get`, `new`, `one` y `body` dominan el lado
contrario, vocabulario conversacional sin tema común.

Los bigramas, en la Figura 2, muestran el valor de subir de nivel: `suicide bomber`,
`suicide bombing`, `oil spill`, `northern california` y `california wildfire` son
expresiones fijas que un unigrama parte en dos piezas sin relación evidente entre sí.
`suicide` por separado y `bomber` por separado dicen menos que la frase completa. `urlweb
urlweb` también aparece entre los primeros lugares de las dos clases: es el artefacto de los
tweets con más de un enlace, no una expresión con significado.

Los trigramas, en la Figura 3, ya no son expresiones generales sino fragmentos casi
literales de un puñado de tweets que se repiten con variaciones mínimas, como `northern
california wildfire` o `severe thunderstorm warning`. Es la contraparte del costo de subir
de nivel: cada trigrama cubre muchos menos tweets que un unigrama, su frecuencia cae rápido,
y el riesgo de que el modelo memorice tweets concretos en vez de aprender un patrón que
generalice crece con cada token adicional.

## Probabilidad condicional de un bigrama

`P(w2 | w1) = count(w1, w2) / count(w1)` es el modelo de bigramas de la referencia de
Jurafsky y Martin que cita el enunciado, calculado aquí sobre las dos clases juntas porque
interesa la estructura del lenguaje y no la separación entre clases. Un bigrama raro cuya
única aparición coincide siempre con la misma palabra siguiente obtiene probabilidad
condicional de 1.0 sin ser una expresión fija real, así que se exige que la primera palabra
aparezca al menos 30 veces en el corpus antes de calcular su probabilidad condicional. Los 10
bigramas con mayor probabilidad condicional están en
`results/tables/ngramas_bigrama_condicional_top10.csv`.

Con ese filtro, la tabla queda dominada por colocaciones genuinas: `oil spill`, `natural
disaster`, `dust storm` y `northern california` tienen probabilidad condicional alta porque,
cuando el corpus usa la primera palabra, casi siempre la sigue exactamente esa segunda
palabra.

## Análisis discriminativo: qué palabras servirán para el modelo

Para cada unigrama se calcula el logaritmo de la razón de probabilidades entre clases:

```
log( P(palabra | desastre) / P(palabra | no_desastre) )
```

con suavizado de Laplace sobre todo el vocabulario y un mínimo de 5 apariciones totales para
evitar que palabras casi únicas dominen la tabla por ruido de muestra pequeña. Un log-ratio
positivo y grande empuja hacia desastre real, uno negativo y grande empuja hacia no desastre.
La tabla completa está en `results/tables/ngramas_discriminativo_completo.csv`, y la Figura 4
muestra las 20 palabras más extremas de cada lado.

Del lado de desastre real el vocabulario es abrumadoramente técnico y específico de eventos:
`mh370` remite al vuelo desaparecido, `hiroshima` al aniversario que coincide con el periodo
de recolección del corpus, `derailment`, `debris`, `wildfire` y `typhoon` son vocabulario de
catástrofe sin uso figurado. Estas son las palabras que un modelo de clasificación debería
aprender a pesar con fuerza hacia la clase positiva.

Del lado de no desastre aparece la confirmación exacta de un hallazgo del análisis
exploratorio inicial: `bag` es la palabra con el log-ratio más negativo de todo el corpus, y
viene casi por completo de la expresión `body bag`, ya señalada como el ejemplo más claro de
vocabulario de catástrofe usado en sentido no literal. `ruin` aparece por la misma razón. Esta
lista es la respuesta con evidencia a qué palabras conviene que el modelo del ejercicio 6
aprenda a distinguir, en lugar de una respuesta basada en intuición.

## Palabras que no discriminan

Un subproducto del análisis discriminativo es identificar palabras frecuentes con log-ratio
cercano a cero: aparecen en las dos clases casi por igual y no ayudan a separarlas. La tabla
está en `results/tables/ngramas_palabras_neutras.csv` y confirma, con una medida distinta, la
misma lección de la selección de stopwords de dominio: frecuencia alta no implica capacidad
de discriminación.

## El caso de `911`

El análisis exploratorio inicial ya había medido que `911` aparece en apenas 4 tweets del
corpus. La verificación sobre `text_clean` lo confirma: al no cumplir el mínimo de 5
apariciones, ni siquiera entra a la tabla discriminativa. Es un caso demasiado pequeño para
sostener una conclusión propia; lo que importa es la señal general de los números, no ese
dígito en particular.

## ¿Vale la pena explorar bigramas o trigramas para analizar contexto?

La evidencia de esta sección responde que sí, con matices y con un costo medible.

A favor de los bigramas: capturan expresiones que un unigrama parte en dos piezas sin
relación evidente, `suicide bomber`, `suicide bombing`, `oil spill`, `california wildfire` y
`natural disaster` son colocaciones reales del corpus, con probabilidad condicional alta una
vez filtrado el ruido de baja frecuencia. `suicide` solo o `bomber` solo dicen menos que la
frase junta, porque cada palabra por separado tiene usos que no son de desastre.

El costo: el vocabulario de bigramas es mucho más disperso que el de unigramas, y los
trigramas llevan ese problema al extremo. Sus términos más frecuentes ya no son expresiones
generales sino fragmentos casi literales de un puñado de tweets repetidos. Cuantos más tokens
tenga el n-grama, menos tweets lo comparten, y mayor es el riesgo de que el modelo memorice
esos tweets concretos en vez de aprender un patrón que generalice.

Recomendación para el modelo preliminar del ejercicio 6: incluir bigramas junto con
unigramas, con un rango de n-gramas de 1 a 2 y una frecuencia mínima de documento que
descarte los bigramas casi únicos, es la combinación con más probabilidad de mejorar sobre
unigramas solos sin pagar el costo de dispersión de los trigramas. Los trigramas quedan como
algo por explorar en la entrega final, condicionado a que el conjunto de entrenamiento sea lo
bastante grande para sostenerlos.
