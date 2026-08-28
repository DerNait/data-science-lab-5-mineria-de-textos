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
| 6 | Modelos de clasificación | **Preliminar** — baseline y 5 combinaciones evaluadas, modelo seleccionado y descrito | `notebooks/04_modelo_preliminar.ipynb` |
| 7 | Función que clasifica un tweet crudo | Pendiente (entrega final) | — |
| 8 | Clasificación en positivo / negativo / neutro | Pendiente (entrega final) | — |
| 9 | Tweets más positivos y más negativos | Pendiente (entrega final) | — |
| 10 | Variable de negatividad y reentrenamiento | **Completo** | `notebooks/08_negatividad_reentreno.ipynb` |
| 11 | Informe | **Versión avance** | `informe/secciones/` |

El alcance actual corresponde a la **entrega de avance** (27 de agosto de 2026), que pide
descripción de los datos, preprocesamiento explicado, unigramas, bigramas y la descripción del
modelo preliminar de clasificación. Los ejercicios 7 a 10 corresponden a la entrega final
(30 de agosto de 2026).

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

Los cuadernos se ejecutan en orden. Cada uno es autónomo: reconstruye el texto procesado que
necesita en lugar de depender de que otro se haya ejecutado antes.

```bash
jupyter lab notebooks/
```

---

## Estructura del repositorio

```
├── codebook.md                      Variables, fuente y limitaciones del conjunto
├── requirements.txt                 Dependencias con versión fija
├── data/
│   ├── raw/train.csv                No se versiona; lo obtiene download_dataset.py
│   └── processed/particion.parquet  Partición congelada train/test
├── src/
│   ├── config.py                    SEED, rutas, umbrales y paleta de colores
│   ├── download_dataset.py          Descarga reproducible y verificación SHA256
│   ├── carga.py                     Lectura, validación y deduplicación
│   └── preprocesamiento.py          Las dos limpiezas y el pipeline de tokens
├── notebooks/
│   ├── 01_carga_y_eda.ipynb         Ejercicios 1, 2 y 5
│   ├── 02_preprocesamiento.ipynb    Ejercicio 3
│   ├── 03_ngramas.ipynb             Ejercicio 4
│   └── 04_modelo_preliminar.ipynb   Ejercicio 6 (preliminar)
├── informe/secciones/               Secciones del informe en Markdown
└── results/
    ├── figures/                     Figuras .png referenciadas en el informe
    ├── tables/                      Tablas .csv de frecuencias y métricas
    └── modelos/                     Modelos serializados
```

---

## Decisiones técnicas del laboratorio

Las decisiones que condicionan todo el análisis, cada una con su justificación en el informe:

- **Solo se usa `train.csv`.** El `test.csv` de la competencia no trae la columna `target`: es
  el conjunto de evaluación ciega de Kaggle. La partición entrenamiento/prueba se obtiene
  dividiendo `train.csv`. Esto explica la discrepancia con las *"más de 10 500 filas"* del
  enunciado, que corresponden a la suma de ambos archivos.
- **Deduplicación antes de partir.** De 7,613 filas se eliminan 128 (18 textos con etiquetas
  contradictorias y 51 duplicados exactos) y quedan **7,485**.
- **Dos preprocesamientos, no uno.** `limpiar_clasificacion` es agresiva y sirve a la tarea de
  tema; `limpiar_sentimiento` es conservadora, mantiene negaciones y signos expresivos, y sirve
  a la tarea de opinión de la entrega final. Aplicar una sola a ambas rompería una de las dos.
- **Las URLs no se borran, se reemplazan por `urlweb`.** Su sola presencia es el patrón más
  discriminativo del corpus (67.35% en desastre contra 41.73% en el resto).
- **Los números se conservan.** Aparecen en el 72.74% de los tweets de desastre contra el
  49.90% del resto. Sobre el `911` que menciona el enunciado: aparece en solo 4 tweets, es
  anecdótico.
- **Métrica de selección: macro-F1.** El desbalance 57.41% / 42.59% hace que un clasificador
  trivial alcance 57% de exactitud sin aprender nada.
- **Sin fuga de información.** El vectorizador va siempre dentro de un `Pipeline` y se ajusta
  solo con el conjunto de entrenamiento.
- **Semilla fija `SEED = 42`** en `src/config.py`, importada por todos los cuadernos.

---

## Resultado del modelo preliminar

| | macro-F1 | Exactitud |
|---|---|---|
| Baseline (clase mayoritaria) | 0.3646 | 57.38% |
| **TF-IDF (1,2) + Regresión logística** | **0.7857** | **79.23%** |

Dos hipótesis del análisis exploratorio se pusieron a prueba en el ejercicio 6:

- **`keyword` como característica del modelo: descartada.** Empeora la macro-F1 en 0.0187,
  porque la keyword ya aparece dentro del texto del tweet en el 94.10% de los casos y el
  one-hot solo añade 222 columnas dispersas.
- **Los errores se concentran en el lenguaje figurado: confirmada.** La tasa de error es del
  25.03% en keywords de tasa intermedia frente al 10.34% en las de tasa extrema alta.

---

## Fuente de los datos

Competencia de Kaggle [*Natural Language Processing with Disaster
Tweets*](https://www.kaggle.com/competitions/nlp-getting-started/data). Los tweets fueron
recolectados por Figure Eight y distribuidos en su colección *Data For Everyone*; cada uno fue
etiquetado manualmente por anotadores humanos. Los detalles están en `codebook.md`.
