# Laboratorio 5 — Clasificación de tweets usando minería de texto

**CC3084 · Ciencia de Datos · Universidad del Valle de Guatemala · Semestre II, 2026**

Clasificación de tweets del conjunto *Natural Language Processing with Disaster Tweets*
(Kaggle) según se refieran o no a un desastre real, mediante minería de texto y análisis de
sentimiento.

---

## Estado por ejercicio

| # | Ejercicio | Estado | Dónde está |
|---|---|---|---|
| 1 | Descargar `train.csv` | **Completo** | `src/download_dataset.py` |
| 2 | Cargar los datos en Python | **Completo** | `src/carga.py`, `notebooks/01_carga_y_eda.ipynb` |
| 3 | Limpieza y preprocesamiento documentado | **Completo** | `src/preprocesamiento.py`, `notebooks/02_preprocesamiento.ipynb` |
| 4 | Frecuencias por clase, n-gramas y probabilidades | **Completo** | `notebooks/03_ngramas.ipynb` |
| 5 | Análisis exploratorio, nubes de palabras, histogramas | **Completo** | `notebooks/01_carga_y_eda.ipynb` |
| 6 | Modelos de clasificación y selección del mejor | **Completo** — 14 combinaciones, 5 variantes de contexto, validación cruzada y análisis de errores | `src/modelos.py`, `notebooks/05_modelos_completos.ipynb` |
| 7 | Función que clasifica un tweet crudo | **Completo** — importable, con casos borde y 12 tweets de prueba propios | `src/clasificador.py`, `notebooks/06_funcion_clasificacion.ipynb` |
| 8 | Clasificación en positivo / negativo / neutro | **Completo** | `src/sentimiento.py`, `notebooks/07_sentimiento.ipynb` |
| 9 | Tweets más positivos y más negativos | **Completo** | `notebooks/07_sentimiento.ipynb` |
| 10 | Variable de negatividad y reentrenamiento | **Completo** | `notebooks/08_negatividad_reentreno.ipynb` |
| 11 | Informe | **Completo** | `informe/informe_final.md` y `informe/Laboratorio5_Informe.pdf` |

---

## Cómo reproducir el análisis

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt_tab'); nltk.download('vader_lexicon')"

python src/download_dataset.py     # descarga y verifica train.csv
```

`download_dataset.py` requiere credenciales de Kaggle en `~/.kaggle/kaggle.json` y haber
aceptado las reglas de la competencia desde el navegador. Si la descarga automática falla, el
script imprime las instrucciones para bajar el archivo a mano. En ambos casos **verifica la
huella SHA256** del CSV, de modo que todo el equipo trabaje sobre exactamente el mismo insumo:

```
61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df
```

Los cuadernos se ejecutan en orden:

```bash
jupyter lab notebooks/
```

Solo dos producen archivos que los demás necesitan, y por eso el orden importa entre ellos:

- **`05_modelos_completos.ipynb`** congela `data/processed/particion.parquet` (si no existe) y
  guarda `results/modelos/mejor_modelo.joblib`, que es lo que carga la función del ejercicio 7.
- **`07_sentimiento.ipynb`** produce `data/processed/tweets_con_sentimiento.parquet`, que es lo
  que consume el reentrenamiento del ejercicio 10.

Ninguno de esos archivos se versiona: son derivados reproducibles del CSV crudo.

Para armar el informe completo a partir de las secciones:

```bash
python informe/ensamblar.py        # genera informe_final.md, .html y el PDF
```

---

## Estructura del repositorio

```
├── codebook.md                        Variables, fuente y limitaciones del conjunto
├── requirements.txt                   Dependencias con versión fija
├── data/
│   ├── raw/train.csv                  No se versiona; lo obtiene download_dataset.py
│   └── processed/                     Partición congelada y corpus anotado (derivados)
├── src/
│   ├── config.py                      SEED, rutas, umbrales y paleta de colores
│   ├── download_dataset.py            Descarga reproducible y verificación SHA256
│   ├── carga.py                       Lectura, validación y deduplicación
│   ├── preprocesamiento.py            Las dos limpiezas y el pipeline de tokens
│   ├── sentimiento.py                 VADER, TextBlob y la variable de negatividad
│   ├── modelos.py                     Partición congelada, pipelines y evaluación
│   └── clasificador.py                clasificar_tweet(): la función del ejercicio 7
├── notebooks/
│   ├── 01_carga_y_eda.ipynb           Ejercicios 1, 2 y 5
│   ├── 02_preprocesamiento.ipynb      Ejercicio 3
│   ├── 03_ngramas.ipynb               Ejercicio 4
│   ├── 04_modelo_preliminar.ipynb     Ejercicio 6 (versión del avance)
│   ├── 05_modelos_completos.ipynb     Ejercicio 6 (parrilla completa y selección)
│   ├── 06_funcion_clasificacion.ipynb Ejercicio 7
│   ├── 07_sentimiento.ipynb           Ejercicios 8 y 9
│   └── 08_negatividad_reentreno.ipynb Ejercicio 10
├── informe/
│   ├── secciones/                     Secciones del informe en Markdown
│   ├── ensamblar.py                   Une las secciones, numera las figuras y arma el PDF
│   └── Laboratorio5_Informe.pdf       Informe final
└── results/
    ├── figures/                       Figuras .png referenciadas en el informe
    ├── tables/                        Tablas .csv de frecuencias y métricas
    └── modelos/                       Modelos serializados (no se versionan)
```

---

## Uso de la función de clasificación

```python
import sys; sys.path.insert(0, 'src')
from clasificador import clasificar_tweet, clasificar_tweets

clasificar_tweet("Massive earthquake just hit downtown, buildings collapsed")
# {'texto_original': 'Massive earthquake just hit downtown, buildings collapsed',
#  'texto_procesado': 'massive earthquake hit downtown building collapsed',
#  'prediccion': 'desastre',
#  'confianza': 0.7998,
#  'sentimiento': 'negativo',
#  'advertencia': None}

clasificar_tweets(["...", "..."])   # DataFrame, una fila por tweet
```

Recibe el tweet **sin preprocesar**: la limpieza completa ocurre dentro. Carga el modelo desde
`results/modelos/mejor_modelo.joblib`, así que funciona en un intérprete limpio sin haber
ejecutado ningún cuaderno. No lanza excepciones por el contenido del texto: los casos borde
—cadena vacía, `None`, solo un enlace, otro idioma, texto más largo que un tweet— devuelven una
predicción y explican en `advertencia` por qué desconfiar de ella.

---

## Decisiones técnicas del laboratorio

Las decisiones que condicionan todo el análisis, cada una con su justificación en el informe:

- **Solo se usa `train.csv`.** El `test.csv` de la competencia no trae la columna `target`: es
  el conjunto de evaluación ciega de Kaggle. La partición entrenamiento/prueba se obtiene
  dividiendo `train.csv`. Esto explica la discrepancia con las *"más de 10 500 filas"* del
  enunciado, que corresponden a la suma de ambos archivos.
- **Deduplicación antes de partir.** De 7,613 filas se eliminan 128 (18 textos con etiquetas
  contradictorias y 51 duplicados exactos) y quedan **7,485**.
- **Partición congelada en disco.** 5,988 tweets de entrenamiento y 1,497 de prueba, guardados
  como lista de `id` en `particion.parquet`. Todos los ejercicios leen esa partición en lugar de
  recalcularla, para que sus resultados sean comparables entre sí.
- **Dos preprocesamientos, no uno.** `limpiar_clasificacion` es agresiva y sirve a la tarea de
  tema; `limpiar_sentimiento` es conservadora, mantiene negaciones y signos expresivos, y sirve
  a la tarea de opinión. Aplicar una sola a ambas rompería una de las dos.
- **Las URLs no se borran, se reemplazan por `urlweb`.** Su sola presencia es el patrón más
  discriminativo del corpus (67.35% en desastre contra 41.73% en el resto).
- **Los números se conservan.** Aparecen en el 72.74% de los tweets de desastre contra el
  49.90% del resto. Sobre el `911` que menciona el enunciado: aparece en solo 4 tweets.
- **Métrica de selección: macro-F1**, medida en validación cruzada sobre el conjunto de
  entrenamiento. El conjunto de prueba no participa en la selección del modelo.
- **Sin fuga de información.** El vectorizador va siempre dentro de un `Pipeline` y se ajusta
  solo con el conjunto de entrenamiento.
- **Semilla fija `SEED = 42`** en `src/config.py`, importada por todos los cuadernos.

---

## Resultados

**Modelo seleccionado:** TF-IDF `(1,2)` con `min_df=2` + regresión logística con
`class_weight='balanced'`, sobre texto limpio y lematizado.

| | macro-F1 | Exactitud | Recall desastre |
|---|---|---|---|
| Baseline (clase mayoritaria) | 0.3646 | 57.38% | 0.0000 |
| **Modelo seleccionado** | **0.7857** | **79.23%** | **0.7241** |

Elegido entre 14 combinaciones de la parrilla (3 algoritmos × 2 representaciones × 2
reducciones, más Random Forest) y 5 variantes de contexto. **Los seis mejores candidatos caen
dentro de una desviación estándar en validación cruzada**, así que la elección de algoritmo y
representación resultó no ser determinante.

Cuatro hipótesis puestas a prueba, y ninguna de las tres primeras sobrevivió:

- **`keyword` como característica:** −0.0187 de macro-F1. La keyword ya aparece dentro del texto
  del tweet en el 94.10% de los casos.
- **Variable de negatividad (ejercicio 10):** −0.0098 de macro-F1, sin significancia en la
  prueba de McNemar (p = 0.1418). Polaridad y factualidad son dimensiones distintas: el 42.22%
  de los tweets de desastre real **no** son negativos.
- **Bigramas y trigramas:** +0.0001 y +0.0036. El 46.6% del vocabulario son bigramas y **ninguno**
  aparece entre los 20 términos más influyentes.
- **Los errores se concentran en el lenguaje figurado: confirmada.** Tasa de error del 25.03% en
  keywords de tasa intermedia contra 10.34% en las de tasa extrema alta.

---

## Fuente de los datos

Competencia de Kaggle [*Natural Language Processing with Disaster
Tweets*](https://www.kaggle.com/competitions/nlp-getting-started/data). Los tweets fueron
recolectados por Figure Eight y distribuidos en su colección *Data For Everyone*; cada uno fue
etiquetado manualmente por anotadores humanos. Los detalles están en `codebook.md`.
