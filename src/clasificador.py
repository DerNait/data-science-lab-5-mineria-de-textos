"""Clasificación de tweets nuevos, sin preprocesar.

Este módulo es el ejercicio 7. El requisito de la rúbrica es literal: una función que
*"recibe el texto de un tweet **sin preprocesar** y devuelve si se refiere a un desastre
natural o no"*.

    clasificar_tweet("Massive earthquake just hit downtown, buildings collapsed")
    -> {'texto_original': ...,
        'texto_procesado': 'massive earthquake hit downtown building collapsed',
        'prediccion': 'desastre',
        'confianza': 0.93,
        'sentimiento': 'negativo',
        'advertencia': None}


Las tres decisiones de diseño
-----------------------------
**1. Todo el preprocesamiento ocurre dentro.** Quien llama a la función escribe un tweet
tal como lo escribiría en Twitter —con URLs, hashtags, mayúsculas y erratas— y no necesita
saber que existen stopwords, lematización ni TF-IDF. Si el preprocesamiento quedara fuera,
cada persona que use el modelo tendría que reproducirlo, y cualquier diferencia mínima
—una stopword de más, el orden de dos pasos— produciría predicciones distintas sin ningún
aviso. Ese es el error clásico de despliegue de un modelo de texto y aquí se evita por
construcción.

**2. El modelo se carga desde el `.joblib`, no de un cuaderno.** El módulo funciona en un
intérprete limpio, sin haber ejecutado nada antes. Se carga una sola vez y se reutiliza.

**3. Los casos borde devuelven una predicción y una advertencia, no una excepción.** Un
tweet vacío, uno que queda vacío tras limpiar, o uno escrito en otro idioma son entradas
reales. La función responde igual —el modelo siempre puede predecir— y añade en
`advertencia` el motivo por el que ese resultado merece desconfianza. Devolver una
excepción obligaría a quien llama a envolver cada uso en un `try`; devolver una predicción
silenciosamente equivocada sería peor.


Sobre la confianza
------------------
El modelo seleccionado es una regresión logística, así que `confianza` es su probabilidad
estimada para la clase predicha. Es una probabilidad del modelo, no una garantía: el
análisis de errores del ejercicio 6 muestra casos con confianza superior a 0.90 que están
mal clasificados. Se reporta porque ordena bien —los errores tienden a concentrarse en las
confianzas bajas— y porque permite filtrar cuando el uso lo requiera.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from preprocesamiento import pipeline_texto  # noqa: E402
from sentimiento import etiquetar_polaridad, puntuar_vader, texto_para_vader  # noqa: E402

RUTA_MODELO = config.MODELOS / "mejor_modelo.joblib"

# Un tweet tiene como máximo 280 caracteres. Por encima de eso el texto no es un tweet, y
# aunque el modelo puede puntuarlo, se avisa: el vocabulario se aprendió sobre textos cortos.
LARGO_MAXIMO_TWEET = 280

# Si menos de esta fracción de los tokens del tweet está en el vocabulario del modelo, la
# predicción se apoya en muy poca evidencia. Suele pasar con texto en otro idioma.
#
# El umbral está calibrado sobre los 1,497 tweets del conjunto de prueba, no elegido a ojo:
# su cobertura mediana es del 83% y solo el 1.87% de ellos queda por debajo de 0.40, así que
# el aviso casi nunca molesta con texto legítimo. Un tweet en español de longitud normal
# ronda el 30% —lo poco que coincide son palabras cortas que también existen en inglés— y sí
# queda avisado.
COBERTURA_MINIMA = 0.40

# Marcadores que el preprocesamiento inserta en lugar de contenido: `urlweb` sustituye a
# cualquier URL y los `emoji_*` a los emoticones. Un texto que **solo** contiene marcadores
# no tiene contenido propio, aunque la cadena no esté vacía. Distinguirlo importa: `urlweb`
# es uno de los términos que más empujan hacia "desastre" (coeficiente 2.26), así que un
# tweet que sea únicamente un enlace se clasifica como desastre con alta confianza sin que
# ninguna palabra lo justifique.
MARCADORES = {"urlweb", "emoji_feliz", "emoji_triste", "emoji_amor"}

_MODELO = None


class ModeloNoDisponible(Exception):
    """No se encontró el modelo entrenado en disco."""


def cargar_modelo(ruta: Path | None = None, recargar: bool = False):
    """Carga el `Pipeline` entrenado y lo deja en memoria para las llamadas siguientes.

    El objeto serializado incluye el vectorizador ya ajustado, con su vocabulario y sus
    pesos IDF. Por eso basta con él: no hay que reconstruir nada.
    """
    global _MODELO
    ruta = Path(ruta) if ruta is not None else RUTA_MODELO

    if _MODELO is None or recargar:
        if not ruta.exists():
            raise ModeloNoDisponible(
                f"No existe {ruta}.\n"
                f"El modelo se genera ejecutando notebooks/05_modelos_completos.ipynb, "
                f"que lo entrena y lo guarda con joblib.")
        _MODELO = joblib.load(ruta)
    return _MODELO


def _vectorizador(modelo):
    """Devuelve el vectorizador del pipeline, esté suelto o dentro de un ColumnTransformer."""
    if "vec" in modelo.named_steps:
        return modelo.named_steps["vec"]
    return modelo.named_steps["features"].named_transformers_["texto"]


def _revisar(texto_original: str, texto_procesado: str, modelo) -> str | None:
    """Devuelve la advertencia que corresponda al caso, o None si no hay ninguna."""
    if not texto_original.strip():
        return ("El texto recibido está vacío. La predicción refleja únicamente el sesgo "
                "del modelo hacia la clase mayoritaria, no el contenido de ningún tweet.")

    if len(texto_original) > LARGO_MAXIMO_TWEET:
        return (f"El texto tiene {len(texto_original)} caracteres y un tweet admite "
                f"{LARGO_MAXIMO_TWEET}. El modelo se entrenó sobre textos cortos.")

    tokens = texto_procesado.split()

    if not tokens:
        return ("El texto quedó vacío después de limpiarlo: era solo menciones, signos o "
                "palabras vacías. No queda contenido sobre el que decidir.")

    if set(tokens) <= MARCADORES:
        return ("Después de limpiarlo, el tweet no contiene ninguna palabra propia: solo "
                "marcadores de enlace o de emoticón. La predicción se apoya únicamente en "
                "ellos y no debe tomarse como una lectura del contenido.")

    vocabulario = _vectorizador(modelo).vocabulary_
    conocidos = [t for t in tokens if t in vocabulario]
    cobertura = len(conocidos) / len(tokens)
    if cobertura < COBERTURA_MINIMA:
        return (f"Solo {len(conocidos)} de {len(tokens)} palabras del tweet están en el "
                f"vocabulario del modelo ({cobertura:.0%}). Puede ser texto en otro idioma, "
                f"jerga o erratas: la predicción se apoya en muy poca evidencia. El corpus "
                f"de entrenamiento es exclusivamente en inglés.")
    return None


def _preparar(texto) -> str:
    """Normaliza la entrada a una cadena. `None` y los nulos de pandas pasan a cadena vacía."""
    if texto is None or (isinstance(texto, float) and pd.isna(texto)):
        return ""
    return str(texto)


def clasificar_tweet(texto, modelo=None) -> dict | pd.DataFrame:
    """Clasifica un tweet crudo, tal como se escribiría en Twitter.

    Args:
        texto: el texto del tweet **sin preprocesar**. También acepta una lista, tupla o
            Serie de tweets; en ese caso devuelve un DataFrame con una fila por tweet.
        modelo: pipeline ya cargado. Por omisión usa el de `results/modelos/`.

    Returns:
        Un diccionario con seis claves:

        - `texto_original`: lo que se recibió, sin tocar.
        - `texto_procesado`: el texto tras la limpieza, para poder auditar la decisión.
        - `prediccion`: `'desastre'` o `'no desastre'`.
        - `confianza`: probabilidad estimada de la clase predicha, en [0.5, 1].
        - `sentimiento`: `'positivo'`, `'negativo'` o `'neutro'` según VADER.
        - `advertencia`: motivo por el que desconfiar del resultado, o `None`.

    Nunca lanza excepción por el contenido del texto. La única excepción posible es
    `ModeloNoDisponible`, si no se ha entrenado el modelo todavía.
    """
    if isinstance(texto, (list, tuple, pd.Series)):
        return clasificar_tweets(texto, modelo=modelo)

    modelo = modelo if modelo is not None else cargar_modelo()
    original = _preparar(texto)

    # El preprocesamiento completo, adentro. `quitar_numeros=False` porque el análisis
    # exploratorio midió que los dígitos son el segundo patrón más discriminativo del
    # corpus: aparecen en el 72.74% de los tweets de desastre contra el 49.90% del resto.
    procesado = " ".join(pipeline_texto(original, modo="clasificacion",
                                        reduccion="lema", quitar_numeros=False))

    etiqueta = int(modelo.predict([procesado])[0])

    if hasattr(modelo, "predict_proba"):
        probabilidad = float(modelo.predict_proba([procesado])[0][etiqueta])
    else:
        # LinearSVC no da probabilidades, solo distancia al hiperplano. Se transforma para
        # que caiga en el rango correcto; ordena igual, pero no está calibrada.
        import numpy as np
        distancia = float(modelo.decision_function([procesado])[0])
        p = 1.0 / (1.0 + np.exp(-distancia))
        probabilidad = p if etiqueta == 1 else 1.0 - p

    compound = puntuar_vader(texto_para_vader(original))["compound"]

    return {
        "texto_original": original,
        "texto_procesado": procesado,
        "prediccion": config.ETIQUETAS[etiqueta],
        "confianza": round(probabilidad, 4),
        "sentimiento": etiquetar_polaridad(compound),
        "advertencia": _revisar(original, procesado, modelo),
    }


def clasificar_tweets(textos, modelo=None) -> pd.DataFrame:
    """Clasifica una colección de tweets crudos y devuelve un DataFrame.

    Una fila por tweet, con las mismas seis columnas que devuelve `clasificar_tweet`.
    El modelo se carga una sola vez para toda la colección.
    """
    modelo = modelo if modelo is not None else cargar_modelo()
    tabla = pd.DataFrame([clasificar_tweet(t, modelo=modelo) for t in textos])

    # pandas convierte los None de una columna mixta en NaN. Se revierte para que la
    # columna signifique lo mismo aquí que en el diccionario: sin advertencia es None.
    tabla["advertencia"] = tabla["advertencia"].astype(object).where(
        tabla["advertencia"].notna(), None)
    return tabla


if __name__ == "__main__":
    ejemplos = [
        "Massive earthquake just hit downtown, buildings collapsed everywhere #prayforus",
        "This exam was a complete disaster, I'm never studying again lol",
        "",
        "http://t.co/abc123 @someone",
    ]
    for resultado in clasificar_tweets(ejemplos).to_dict("records"):
        print(f"[{resultado['prediccion']:>11}] conf={resultado['confianza']:.3f} "
              f"sent={resultado['sentimiento']:>8}  {resultado['texto_original'][:60]!r}")
        if resultado["advertencia"]:
            print(f"              aviso: {resultado['advertencia'][:90]}")
