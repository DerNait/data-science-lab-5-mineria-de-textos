# Respuesta a las interrogantes

Esta sección responde las tres preguntas del **ejercicio 9**. Todas se apoyan en la puntuación
de VADER descrita en la sección anterior, calculada sobre el texto con limpieza conservadora y
emoticones literales. El trabajo está en `notebooks/07_sentimiento.ipynb` y las tablas en
`results/tables/sentimiento_top10_*.csv` y `sentimiento_comparacion_clases.csv`.

## 9.1 ¿Cuáles son los 10 tweets más negativos y en qué categoría están?

**Ocho de los diez tweets más negativos son de desastre real.** La tabla completa está en
`results/tables/sentimiento_top10_negativos.csv`.

| # | `compound` | Categoría | Tweet (recortado) |
|---|---|---|---|
| 1 | −0.9883 | no desastre | *"wreck? wreck wreck wreck wreck wreck wreck..."* |
| 2 | −0.9686 | **desastre** | *"Suicide bomber targets Saudi mosque at least 13 dead..."* |
| 3 | −0.9623 | **desastre** | *"Suicide bomber kills 15 in Saudi security site mosque..."* |
| 4 | −0.9595 | **desastre** | *"Nigeria: Suicide Bomb Attacks Killed 64 People; Blamed: Boko Haram"* |
| 5 | −0.9552 | **desastre** | *"17 killed in S. Arabia mosque suicide bombing"* |
| 6 | −0.9549 | no desastre | *"at the lake \*sees a dead fish\* me: poor little guy..."* |
| 7 | −0.9538 | **desastre** | *"illegal alien... Charged With Rape & Murder of Santa Maria CA Woman"* |
| 8 | −0.9524 | **desastre** | *"Bomb Crash Loot Riot Emergency Pipe Bomb Nuclear Chemical Spill..."* |
| 9 | −0.9500 | **desastre** | *"Bomb head? Explosive decisions dat produced more dead children..."* |
| 10 | −0.9493 | **desastre** | *"...you are the biggest terrorist and trouble maker in the world"* |

El contenido de los ocho es inequívoco: atentados suicidas en Arabia Saudita y Nigeria,
muertes, bombas. Es vocabulario de catástrofe usado en sentido literal, y VADER lo puntúa
correctamente por debajo de −0.95.

Los dos que **no** son de desastre son instructivos. El primero, y el más negativo de todo el
corpus, es la repetición literal de la palabra `wreck` trece veces. VADER acumula la valencia
de cada repetición y llega casi al extremo de la escala, pero el tweet no reporta ningún
evento: **es un artefacto de la aritmética del léxico**, que suma ocurrencias sin advertir que
se trata de la misma palabra repetida. El segundo describe un pez muerto en un lago, un suceso
menor narrado con lenguaje dramático.

Ambos apuntan a lo mismo: **negatividad extrema no equivale a desastre real**, ni siquiera en
el extremo de la distribución donde la asociación es más fuerte.

## 9.2 ¿Cuáles son los 10 tweets más positivos y en qué categoría están?

**Ocho de los diez tweets más positivos no son de desastre real**, el espejo casi exacto del
resultado anterior. La tabla está en `results/tables/sentimiento_top10_positivos.csv`.

| # | `compound` | Categoría | Tweet (recortado) |
|---|---|---|---|
| 1 | +0.9730 | no desastre | *"Want Twister Tickets AND A VIP EXPERIENCE To See SHANIA?"* |
| 2 | +0.9564 | no desastre | *"...I'm happy you've survived all of them. Hope you're okay"* |
| 3 | +0.9471 | **desastre** | *"Today's storm will pass; let tomorrow's light greet you with a kiss..."* |
| 4 | +0.9423 | no desastre | *"we enjoyed the show today. Great fun. The emergency non evacuation..."* |
| 5 | +0.9376 | no desastre | *"Free Ebay Sniping... Lumbar Extender Back Stretcher Excellent Condition!!"* |
| 6 | +0.9356 | **desastre** | *"`:)` well I think that sounds like a fine plan where little derailment is possible"* |
| 7 | +0.9345 | no desastre | *"I'm not a Drake fan but I enjoy seeing him body-bagging people. Great marketing lol."* |
| 8 | +0.9344 | no desastre | *"yeah we survived 9 seasons and 2 movies. Let's hope for..."* |
| 9 | +0.9300 | no desastre | *"Super sweet and beautiful `:)`"* |
| 10 | +0.9287 | no desastre | *"...got deluge of divine blessing 4 happy n peaceful life"* |

Los ocho son promociones, entusiasmo por una serie, canciones y agradecimientos: registro
conversacional cotidiano.

Los dos etiquetados como desastre real merecen atención porque explican algo del conjunto, y
son casos distintos entre sí:

- *"Today's storm will pass; let tomorrow's light greet you with a kiss..."* es un mensaje
  inspiracional que usa `storm` como metáfora, no un reporte de un temporal.
- *"...well I think that sounds like a fine plan where little **derailment** is possible"* usa
  `derailment` en sentido figurado, para hablar de un plan que difícilmente se descarrile. Es
  especialmente llamativo porque el análisis exploratorio había medido que `derailment` tiene
  una **tasa de desastre real del 100%**: era una de las keywords más inequívocas del corpus.
  Aquí aparece el contraejemplo.

Los dos están etiquetados con `target = 1` pese a no describir ningún evento, lo que muy
probablemente sea **ruido de anotación**. Encajan con los 18 textos de etiqueta contradictoria
que la sección de datos ya documentó, y son el mismo fenómeno de uso figurado que atraviesa
todo el laboratorio, esta vez confundiendo a los anotadores humanos y no al modelo.

**Una observación conjunta sobre 9.1 y 9.2.** En el extremo negativo la asociación con desastre
real es fuerte (8 de 10) y en el extremo positivo la asociación con "no desastre" es igual de
fuerte (8 de 10). **El sentimiento separa bien en los extremos de la distribución**, que es
donde vive una minoría del corpus. Lo que la pregunta siguiente examina es si sigue separando
en el centro, que es donde está casi todo.

## 9.3 ¿Son los tweets de desastre real más negativos que los de la otra categoría?

**Sí: los tweets de desastre real son medibles más negativos, la diferencia es estadísticamente
indiscutible y su magnitud es moderada.** El matiz que acompaña esa respuesta es el hallazgo
más importante de esta sección, y condiciona el ejercicio 10.

### La evidencia

| Métrica | No desastre | Desastre real |
|---|---|---|
| `compound` medio | −0.0571 | **−0.2692** |
| `compound` mediana | 0.0000 | **−0.3182** |
| Desviación estándar | 0.4623 | 0.4338 |
| Negatividad media | 0.5285 | **0.6346** |
| Negatividad mediana | 0.5000 | **0.6591** |

La mediana es más elocuente que la media: la de la clase "no desastre" es **exactamente cero**,
es decir el tweet típico que no habla de un desastre no tiene carga de sentimiento en absoluto,
mientras que el típico de desastre está claramente en terreno negativo.

La prueba estadística es **Mann-Whitney U**, elegida porque no asume normalidad —la
distribución del `compound` está lejos de serlo, con una acumulación fuerte en el cero, como se
ve en la **Figura 3**:

```
U = 5,088,009        p ≈ 3.9 × 10⁻⁸²        r (rango-biserial) = 0.2572
```

El valor p descarta el azar bajo cualquier lectura razonable. Pero **el tamaño del efecto,
r = 0.2572, corresponde a un efecto pequeño a moderado**, no grande. Esa es la cifra que hay
que reportar junto al valor p: con 7,485 observaciones un p astronómicamente pequeño acompaña a
diferencias modestas, y confundir significancia estadística con magnitud sería el error
clásico. La conclusión honesta es que **la diferencia es real y no es grande**.

El violín de la **Figura 3** muestra por qué. Las dos distribuciones **se solapan
ampliamente**: la de desastre está desplazada hacia abajo, pero no es un bloque separado, es la
misma forma corrida. Un tweet tomado al azar de la clase desastre es más probablemente
negativo, pero **saber que un tweet es negativo no permite afirmar que sea de desastre**.

### El matiz: más de cuatro de cada diez desastres no son negativos

El reparto de polaridad dentro de cada categoría es lo que más dice:

| Categoría | Negativo | Neutro | Positivo |
|---|---|---|---|
| No desastre | 42.91% | 24.99% | 32.09% |
| Desastre real | **57.78%** | **26.00%** | **16.22%** |

Los dos números decisivos no están en la columna "negativo" sino en las otras dos.
**El 26.00% de los tweets de desastre real son neutros** —reportes periodísticos factuales del
tipo *"13,000 people receive #wildfires evacuation orders in California"*, que informa de una
catástrofe sin una sola palabra cargada— y **un 16.22% son directamente positivos**: rescates,
supervivientes, agradecimientos. En conjunto, **el 42.22% de los tweets de desastre real no son
negativos**.

Al mismo tiempo, **el 42.91% de los tweets que no describen ningún desastre sí son negativos**,
porque la gente tuitea sobre su mal día con lenguaje catastrófico.

### Consecuencia para el ejercicio 10

**Polaridad y factualidad son dimensiones distintas.** Un tweet puede ser negativo sin hablar de
un desastre, y puede reportar un desastre sin ser negativo. Un desastre real cubierto por un
medio se redacta de forma factual y neutra; una queja cotidiana se redacta con el vocabulario
más dramático disponible.

Esto **anticipa que la variable de negatividad aportará poco al clasificador del ejercicio 10**,
y lo anticipa con datos y no con intuición. Respalda además la hipótesis que el análisis
exploratorio ya había formulado por otra vía, al observar que los tweets de desastre contienen
menos exclamaciones y menos menciones y son por tanto más factuales y menos emocionales.

Conviene dejarlo escrito **antes** de ver el resultado del reentrenamiento: si la variable no
mejora el modelo, será un resultado coherente con toda la evidencia de esta sección y no un
fracaso del método.
