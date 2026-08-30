"""Construcción, entrenamiento y evaluación de los modelos de clasificación.

Este módulo es el punto de entrada al ejercicio 6. Su razón de existir es que tres
cuadernos distintos (05 de modelos, 06 de la función de clasificación y 08 del
reentrenamiento con negatividad) tienen que entrenar y comparar sobre **exactamente la
misma partición**, o las comparaciones entre ellos no significan nada.

Contrato de interfaces, cerrado desde el plan de la entrega final:

    construir_pipeline(nombre_modelo, representacion, columnas_numericas=None) -> Pipeline
    evaluar(pipeline, X_train, y_train, X_test, y_test, etiqueta) -> dict
    cargar_particion(seed=42) -> (entrenamiento, prueba)


Por qué la partición se lee de disco y no se recalcula
------------------------------------------------------
`train_test_split` con la misma semilla devuelve la misma partición **solo si recibe las
filas en el mismo orden**. Basta con que alguien filtre el DataFrame, cambie la regla de
deduplicación o reordene por `id` para que la partición se mueva sin que nadie lo note, y
entonces el modelo del ejercicio 10 estaría midiéndose contra un conjunto de prueba
distinto al del ejercicio 6. La comparación quedaría inservible y, peor, lo parecería
válida.

Por eso la partición se congeló una sola vez en `data/processed/particion.parquet`, como
una lista explícita de `id` con el lado que le tocó a cada uno, y desde entonces se **lee**.
`cargar_particion()` la reconstruye solo si el archivo no existe.


Sin fuga de información
-----------------------
El vectorizador va siempre dentro del `Pipeline`, nunca ajustado por separado sobre el
conjunto completo. Cuando `pipeline.fit` se ejecuta, el vocabulario y los pesos IDF se
calculan usando únicamente el conjunto de entrenamiento; las palabras que solo aparecen
en el conjunto de prueba no forman parte del vocabulario, que es justo lo que pasaría con
un tweet nuevo en producción.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from carga import cargar_tweets  # noqa: E402
from preprocesamiento import pipeline_texto  # noqa: E402

# El corpus con el texto ya procesado se cachea: aplicar el pipeline de limpieza a 7,485
# tweets dos veces (lematizado y con stemming) toma cerca de un minuto, y los cuadernos 05
# y 06 lo necesitan igual. El archivo es un derivado reproducible, así que no se versiona.
CORPUS_PROCESADO = config.PROCESSED / "corpus_procesado.parquet"

COLUMNAS_CORPUS = ["id", "text", "text_clean", "text_stem", "keyword_norm", "target"]

MEJOR_MODELO = config.MODELOS / "mejor_modelo.joblib"


# ---------------------------------------------------------------------------
# Catálogo de algoritmos y representaciones
# ---------------------------------------------------------------------------
# Se declaran como fábricas y no como instancias para que cada combinación reciba un
# estimador nuevo. Reutilizar una misma instancia entre combinaciones haría que la
# segunda heredara el ajuste de la primera en algunos casos, y el error sería silencioso.

def _logreg():
    return LogisticRegression(max_iter=1000, class_weight="balanced",
                              random_state=config.SEED)


def _svc():
    return LinearSVC(class_weight="balanced", random_state=config.SEED)


def _cnb():
    # ComplementNB es la variante de Naive Bayes pensada para clases desbalanceadas.
    return ComplementNB()


def _mnb():
    return MultinomialNB()


def _rf():
    # Se incluye a sabiendas de que los bosques sufren con matrices muy dispersas: el
    # material de clase lo advierte y el resultado sirve de evidencia en el informe.
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  n_jobs=-1, random_state=config.SEED)


ALGORITMOS = {
    "logreg": _logreg,
    "svc": _svc,
    "cnb": _cnb,
    "mnb": _mnb,
    "rf": _rf,
}

NOMBRES_ALGORITMO = {
    "logreg": "Regresión logística",
    "svc": "SVM lineal",
    "cnb": "ComplementNB",
    "mnb": "MultinomialNB",
    "rf": "Random Forest",
}


def construir_vectorizador(representacion: str, ngram_range: tuple = (1, 2),
                           min_df: int = 2, max_features: int | None = None):
    """Devuelve el vectorizador correspondiente a la representación pedida.

    - `bow`   bolsa de palabras: cuenta cuántas veces aparece cada término.
    - `tfidf` pondera esa cuenta por lo raro que es el término en la colección.
    - `char`  n-gramas de caracteres dentro de los límites de palabra. No conoce el
              concepto de palabra, así que absorbe erratas, plurales y truncamientos,
              que abundan en Twitter.
    """
    if representacion == "bow":
        return CountVectorizer(ngram_range=ngram_range, min_df=min_df,
                               max_features=max_features)
    if representacion == "tfidf":
        return TfidfVectorizer(ngram_range=ngram_range, min_df=min_df,
                               max_features=max_features)
    if representacion == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range,
                               min_df=min_df, max_features=max_features)
    raise ValueError(f"Representación desconocida: {representacion!r}. "
                     f"Use 'bow', 'tfidf' o 'char'.")


def construir_pipeline(nombre_modelo: str, representacion: str,
                       columnas_numericas: list | None = None,
                       columna_texto: str = "text_clean",
                       columnas_categoricas: list | None = None,
                       **opciones) -> Pipeline:
    """Arma el `Pipeline` de una combinación algoritmo + representación.

    Si `columnas_numericas` (o `columnas_categoricas`) no es None, usa un
    `ColumnTransformer` para combinar el texto vectorizado con esas variables, y
    entonces el pipeline espera un DataFrame en lugar de una Serie.

    `opciones` se pasa tal cual al vectorizador: `ngram_range`, `min_df`, `max_features`.
    """
    if nombre_modelo not in ALGORITMOS:
        raise ValueError(f"Algoritmo desconocido: {nombre_modelo!r}. "
                         f"Opciones: {sorted(ALGORITMOS)}")

    vectorizador = construir_vectorizador(representacion, **opciones)
    clasificador = ALGORITMOS[nombre_modelo]()

    if columnas_numericas is None and columnas_categoricas is None:
        return Pipeline([("vec", vectorizador), ("clf", clasificador)])

    transformadores = [("texto", vectorizador, columna_texto)]
    if columnas_categoricas:
        transformadores.append(
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas))
    if columnas_numericas:
        # MinMaxScaler y no StandardScaler: Naive Bayes exige entradas no negativas, y
        # estandarizar produciría valores negativos que lo harían fallar.
        transformadores.append(("numericas", MinMaxScaler(), columnas_numericas))

    return Pipeline([
        ("features", ColumnTransformer(transformadores)),
        ("clf", clasificador),
    ])


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def _describir(pipeline: Pipeline) -> dict:
    """Extrae del pipeline cómo se representó el texto y qué algoritmo se usó."""
    clasificador = pipeline.named_steps.get("clf")
    paso_vec = pipeline.named_steps.get("vec")

    if paso_vec is None and "features" in pipeline.named_steps:
        columnas = pipeline.named_steps["features"].transformers
        paso_vec = next((t[1] for t in columnas if t[0] == "texto"), None)
        representacion = f"{type(paso_vec).__name__} + ColumnTransformer"
    else:
        representacion = type(paso_vec).__name__

    detalle = ""
    if paso_vec is not None and hasattr(paso_vec, "ngram_range"):
        analizador = getattr(paso_vec, "analyzer", "word")
        detalle = f"{analizador} {paso_vec.ngram_range}"

    return {
        "representacion": representacion,
        "detalle": detalle,
        "algoritmo": type(clasificador).__name__,
    }


def evaluar(pipeline: Pipeline, X_train, y_train, X_test, y_test,
            etiqueta: str) -> dict:
    """Entrena, predice y devuelve una fila de métricas.

    La fila incluye la predicción bajo la clave `prediccion`, porque el análisis de
    errores y la prueba de McNemar la necesitan después. `tabla_metricas()` la descarta
    al armar el DataFrame comparativo.
    """
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=[0, 1],
                                                 zero_division=0)
    fila = {"modelo": etiqueta}
    fila.update(_describir(pipeline))
    fila.update({
        "accuracy": accuracy_score(y_test, pred),
        "precision_no_desastre": p[0], "recall_no_desastre": r[0], "f1_no_desastre": f[0],
        "precision_desastre": p[1], "recall_desastre": r[1], "f1_desastre": f[1],
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "prediccion": pred,
        "pipeline": pipeline,
    })
    return fila


def tabla_metricas(filas: list, ordenar: bool = True) -> pd.DataFrame:
    """Arma el DataFrame comparativo a partir de las filas que devuelve `evaluar`."""
    tabla = pd.DataFrame([{k: v for k, v in fila.items()
                           if k not in ("prediccion", "pipeline")} for fila in filas])
    if ordenar:
        tabla = tabla.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return tabla


def puntuar_confianza(pipeline: Pipeline, X) -> np.ndarray:
    """Devuelve la confianza en la clase `desastre`, en [0, 1].

    Los modelos probabilísticos devuelven su probabilidad. `LinearSVC` no tiene una: solo
    una distancia con signo al hiperplano. En ese caso se aplica una función logística a
    esa distancia, lo que produce un número en el rango correcto y **ordena igual que la
    distancia**, pero no es una probabilidad calibrada y no debe leerse como tal.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    distancia = pipeline.decision_function(X)
    return 1.0 / (1.0 + np.exp(-distancia))


# ---------------------------------------------------------------------------
# Corpus y partición
# ---------------------------------------------------------------------------

def preparar_corpus(forzar: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Devuelve el corpus deduplicado con el texto ya procesado.

    Columnas: id, text (crudo), text_clean (lematizado), text_stem, keyword_norm, target.

    `quitar_numeros=False` no es un descuido: el análisis exploratorio midió que los
    dígitos aparecen en el 72.74% de los tweets de desastre contra el 49.90% del resto, así
    que borrarlos destruiría el segundo patrón más discriminativo del corpus.
    """
    if CORPUS_PROCESADO.exists() and not forzar:
        corpus = pd.read_parquet(CORPUS_PROCESADO)
        if verbose:
            print(f"Corpus procesado leído de {CORPUS_PROCESADO.name}: {len(corpus):,} tweets")
        return corpus

    df = cargar_tweets(deduplicar=True, verbose=verbose)
    if verbose:
        print("Procesando el texto (esto toma alrededor de un minuto)...")

    df["text_clean"] = df["text"].apply(
        lambda t: " ".join(pipeline_texto(t, modo="clasificacion", reduccion="lema",
                                          quitar_numeros=False)))
    df["text_stem"] = df["text"].apply(
        lambda t: " ".join(pipeline_texto(t, modo="clasificacion", reduccion="stem",
                                          quitar_numeros=False)))
    # `keyword` trae los espacios codificados como %20 en el archivo original.
    df["keyword_norm"] = df["keyword"].fillna("sin_keyword").map(
        lambda k: unquote(str(k)).lower())

    corpus = df[COLUMNAS_CORPUS].copy()
    config.asegurar_directorios()
    corpus.to_parquet(CORPUS_PROCESADO, index=False)
    if verbose:
        print(f"Corpus procesado guardado en {CORPUS_PROCESADO.name}: {len(corpus):,} tweets")
    return corpus


def congelar_particion(corpus: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Crea la partición estratificada y la escribe en disco. Se ejecuta una sola vez."""
    entrenamiento, prueba = train_test_split(
        corpus, test_size=config.TEST_SIZE, stratify=corpus["target"], random_state=seed)

    particion = corpus[["id", "target"]].copy()
    particion["conjunto"] = np.where(particion["id"].isin(entrenamiento["id"]),
                                     "entrenamiento", "prueba")
    config.asegurar_directorios()
    particion.to_parquet(config.PARTICION, index=False)
    return particion


def cargar_particion(seed: int = config.SEED, verbose: bool = True) -> tuple:
    """Devuelve SIEMPRE la misma partición `(entrenamiento, prueba)`.

    Ambos son DataFrames con las columnas de `COLUMNAS_CORPUS`. La partición se lee del
    parquet congelado; solo se recalcula si el archivo no existe todavía.
    """
    corpus = preparar_corpus(verbose=verbose)

    if config.PARTICION.exists():
        particion = pd.read_parquet(config.PARTICION)
        if verbose:
            print(f"Partición congelada leída de {config.PARTICION.name}")
    else:
        particion = congelar_particion(corpus, seed=seed)
        if verbose:
            print(f"Partición congelada creada en {config.PARTICION.name}")

    faltantes = set(corpus["id"]) - set(particion["id"])
    if faltantes:
        raise ValueError(
            f"La partición congelada no cubre {len(faltantes)} tweets del corpus. "
            f"Se generó con una regla de deduplicación distinta: hay que regenerarla.")

    lados = corpus.merge(particion[["id", "conjunto"]], on="id", validate="one_to_one")
    entrenamiento = lados[lados.conjunto == "entrenamiento"].drop(columns="conjunto")
    prueba = lados[lados.conjunto == "prueba"].drop(columns="conjunto")

    if verbose:
        print(f"Entrenamiento: {len(entrenamiento):,} tweets   "
              f"Prueba: {len(prueba):,} tweets")

    return entrenamiento.reset_index(drop=True), prueba.reset_index(drop=True)


if __name__ == "__main__":
    tr, te = cargar_particion()
    print()
    print("Proporción de clases:")
    print(pd.DataFrame({
        "entrenamiento": tr.target.value_counts(normalize=True).sort_index() * 100,
        "prueba": te.target.value_counts(normalize=True).sort_index() * 100,
    }).round(2).to_string())
