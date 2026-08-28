# Variable de negatividad y reentrenamiento

Esta sección responde el **ejercicio 10**: construir una variable que capture la negatividad
de cada tweet, incorporarla al modelo de clasificación y responder si esa inclusión mejora los
resultados. El trabajo completo está en `notebooks/08_negatividad_reentreno.ipynb`, sobre la
misma partición congelada del ejercicio 6 y el mismo corpus anotado con VADER de la sección
anterior.

## La variable: definición y distribución

La negatividad se define a partir del `compound` de VADER, ya descrito y calculado en la
sección anterior:

```
negatividad = (1 - compound) / 2
```

Es una transformación lineal que lleva el rango de VADER, de −1 a 1, al rango 0 a 1. Un tweet
máximamente positivo obtiene negatividad 0, uno máximamente negativo obtiene negatividad 1 y
uno neutro queda en 0.5. Se define en ese rango a propósito: es el mismo en el que produce sus
valores el vectorizador TF-IDF, así que puede concatenarse a la matriz de características sin
necesidad de escalarla.

Sobre las 7,485 filas del corpus, la negatividad tiene media 0.5737 y desviación estándar
0.2312, con valores que van de 0.0135 a 0.9942. La **Figura 1** muestra la distribución
completa y la distribución separada por clase, guardada en
`results/figures/negatividad_histograma.png`. La sección anterior ya había medido la
negatividad media por clase: 0.5285 en los tweets que no son de desastre y 0.6346 en los que sí
lo son, una diferencia consistente con lo que muestra la figura, aunque con un solapamiento
amplio entre ambas distribuciones.

## Relación con la clase, medida antes de entrenar

Antes de meter la variable a ningún modelo conviene saber qué tan fuerte es su relación con
`target`. La correlación punto-biserial entre `negatividad` y `target` es 0.2268, con un valor
p menor a 0.0001; la correlación de Spearman da un resultado prácticamente igual, 0.2219,
también significativa. La tabla completa está en `results/tables/negatividad_correlacion.csv`.

No es una correlación nula, pero es débil. Los tweets de desastre real tienden a ser algo más
negativos, en la misma dirección que ya había encontrado la sección anterior, pero la relación
está lejos de bastar por sí sola para separar las clases. Con una correlación de esta magnitud
la expectativa, escrita antes de correr el reentrenamiento, es que agregar la variable a un
modelo que ya lee el texto completo aporte poco: buena parte de la señal que produce esa
correlación ya está contenida, de forma implícita, en las palabras del tweet.

## El modelo base: prueba de control

El reentrenamiento tiene sentido solo si parte exactamente del mismo modelo y la misma
partición que ya seleccionó el ejercicio 6. Ese modelo es TF-IDF con unigramas y bigramas más
regresión logística, entrenado sobre `text_clean`, guardado en
`results/modelos/modelo_preliminar.joblib`. Cargarlo y evaluarlo sobre el conjunto de prueba de
esta sección da una macro-F1 de 0.7857, el mismo valor, cifra por cifra, que había reportado el
ejercicio 6. La prueba de control queda superada: la partición y el pipeline se están
reproduciendo con exactitud, y cualquier diferencia que aparezca a partir de aquí se debe a la
variable agregada y no a un cambio en los datos.

## El reentrenamiento: negatividad y las cuatro componentes de VADER

Se prueban dos versiones aumentadas, ambas construidas con `ColumnTransformer` para combinar el
mismo vectorizador TF-IDF sobre `text_clean` con columnas numéricas adicionales, y ambas con la
misma regresión logística, la misma partición y la misma semilla que el modelo base. La
diferencia entre las tres versiones es únicamente la matriz de características:

- **Base:** solo TF-IDF sobre el texto.
- **Con negatividad:** TF-IDF más la columna `negatividad`.
- **Con las cuatro componentes de VADER:** TF-IDF más `vader_neg`, `vader_neu`, `vader_pos` y
  `vader_compound`.

| Modelo | Accuracy | Macro-F1 | Diferencia en macro-F1 |
|---|---|---|---|
| Base | 0.7923 | 0.7857 | 0.0000 |
| + negatividad | 0.7822 | 0.7759 | −0.0098 |
| + 4 componentes VADER | 0.7809 | 0.7746 | −0.0111 |

Tabla completa en `results/tables/comparacion_con_sin_negatividad.csv`. Ninguna de las dos
versiones aumentadas mejora al modelo base; ambas empeoran, de forma consistente en precisión y
recall de las dos clases y no como un efecto aislado en una sola métrica. El resultado confirma
la hipótesis planteada en la sección anterior: negatividad y ser un tweet de desastre real son
dimensiones distintas, y agregar la primera no ayuda al modelo a distinguir la segunda. Entre
las dos versiones aumentadas, la variable única se comporta mejor que las cuatro componentes por
separado, algo esperable porque `negatividad` es una función lineal de `compound`, así que sumar
además `neg`, `neu` y `pos` solo añade dimensiones redundantes.

## Prueba de McNemar

Comparando las predicciones del modelo base contra las del mejor modelo aumentado, el de
negatividad, sobre el mismo conjunto de prueba: 38 tweets que el modelo base clasifica mal y el
modelo aumentado clasifica bien, contra 53 que el modelo base clasifica bien y el aumentado
clasifica mal. La prueba binomial exacta de McNemar sobre esos pares discordantes da un valor p
de 0.1418, por encima de 0.05. La diferencia entre los dos modelos no es estadísticamente
distinguible del ruido de muestreo con este tamaño de conjunto de prueba. Aun así, la asimetría
entre 38 y 53 apunta en la misma dirección que ya señalaba la caída de macro-F1: hay más casos
dañados que corregidos. Tabla en `results/tables/negatividad_mcnemar.csv`.

## Dónde cambian las predicciones

Aunque la métrica agregada apenas se mueve, vale la pena mirar qué pasa tweet por tweet. 91 de
los 1,497 tweets de prueba, el 6.08%, cambian de predicción al agregar la variable de
negatividad. De esos, 38 se corrigen y 53 se dañan. Tabla completa en
`results/tables/negatividad_cambios_prediccion.csv`.

Los ejemplos muestran el mecanismo con claridad. El tweet *"tried 11 eye akame ga kill tokyo
ghoul damn bloody dont dare watch"*, sobre una serie de anime, tiene negatividad 0.9527; el
modelo base lo clasifica correctamente como no desastre y el modelo aumentado lo empuja de forma
incorrecta hacia desastre real, guiado por el lenguaje violento y no por el tema. Algo similar
ocurre con *"loner diary pattern sand may blown away photo two choked flame urlweb"*, negatividad
0.8649, dañado en la misma dirección.

En sentido contrario, *"biolab safety concern grow fedex stop transporting certain specimen
research facility"* tiene negatividad baja, 0.2447; el modelo base lo clasificaba de forma
incorrecta como desastre y la baja negatividad ayuda al modelo aumentado a corregirlo. El
patrón es el que anticipó la sección anterior: la negatividad mide tono emocional, no tema, y
cuando el lenguaje figurado hace que un texto suene violento o dramático sin describir un evento
real, la variable empuja la predicción en la dirección equivocada.

## El peso que le da el modelo

El coeficiente de `negatividad` en la regresión logística es 1.6003, frente a 3.1994 del término
de texto con mayor peso del vocabulario. No es un coeficiente despreciable: es aproximadamente
la mitad del término más fuerte, así que el modelo sí le está prestando atención real a la
variable. El problema no es que la ignore, sino que la señal que aporta compite con la del texto
en los casos donde polaridad y tema apuntan en direcciones distintas, y en esos casos el peso
que recibe termina empeorando la predicción en vez de reforzarla. Tabla en
`results/tables/negatividad_coeficientes.csv`.

## Respuesta a la pregunta del enunciado

**No, incluir la variable de negatividad no mejoró los resultados del modelo de clasificación.**
La macro-F1 bajó de 0.7857 a 0.7759 con la versión más favorable, una diferencia de −0.0098, y
la prueba de McNemar no encuentra que esa diferencia sea distinguible del ruido de muestreo, con
un valor p de 0.1418.

La explicación no es que la variable esté mal construida ni que el modelo la ignore: la
correlación entre negatividad y `target` es real, 0.2268, y el coeficiente que recibe dentro del
modelo lineal tiene un peso considerable. La explicación es conceptual. Esa correlación ya está
contenida, de forma implícita, en el texto que procesa el vectorizador de TF-IDF, así que
agregarla como columna aparte no le entrega al modelo información genuinamente nueva, solo una
dimensión adicional en la que fijarse. En los tweets donde el lenguaje figurado hace que un texto
suene negativo sin describir un desastre real, esa dimensión adicional termina empujando la
predicción en la dirección equivocada, y eso pesa un poco más que los casos en los que ayuda.

Es la confirmación, con evidencia medida y no solo con intuición, de la diferencia conceptual
entre polaridad de opinión y tema de desastre que ya había anticipado la sección anterior. Un
tweet puede ser negativo sin hablar de ningún desastre, como una queja sobre un mal día, y uno
sobre un desastre real puede redactarse de forma neutra y factual, como el reporte de un
terremoto de magnitud 6.2. Medir cuánto se enoja o se entristece quien escribe un tweet no es lo
mismo que medir de qué está hablando, y el costo de confundir ambas cosas, aunque pequeño, queda
medido en esta sección.
