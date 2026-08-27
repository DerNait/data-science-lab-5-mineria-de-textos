"""Limpieza y tokenización de tweets, en dos variantes con propósitos distintos.

Este módulo expone:

    limpiar_clasificacion(texto: str) -> str
    limpiar_sentimiento(texto: str) -> str
    tokenizar(texto: str) -> list
    pipeline_texto(texto, modo='clasificacion', reduccion='lema',
                    quitar_numeros=True) -> list

Hay dos limpiezas porque hay dos tareas distintas, y aplicar la misma a ambas
rompe una de las dos. El detalle está en informe/secciones/03_preprocesamiento.md.

- `limpiar_clasificacion`: agresiva. Sirve para el ejercicio 6, que pregunta
  si un tweet describe un desastre real y por tanto es una tarea de TEMA.
  Cuanto más se reduzca el vocabulario a lo temáticamente relevante, mejor
  generaliza el modelo.
- `limpiar_sentimiento`: conservadora. Sirve para los ejercicios 8 a 10 de la
  entrega final. Una tarea de OPINIÓN necesita justo lo que la de tema
  descarta: mayúsculas, signos de puntuación expresivos y, sobre todo,
  negaciones.

Las decisiones de esta limpieza están basadas en el análisis exploratorio de
informe/secciones/02_eda.md. Las URLs y los números son señal discriminativa
y no se descartan a ciegas. El corpus está afectado por mojibake en el 7.90%
de los tweets y no tiene emojis Unicode en absoluto.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# --------------------------------------------------------------------------
# Recursos de NLTK: se descargan una sola vez, de forma perezosa e idempotente.
# --------------------------------------------------------------------------
def _asegurar_recursos_nltk() -> None:
    recursos = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for ruta, paquete in recursos.items():
        try:
            nltk.data.find(ruta)
        except LookupError:
            nltk.download(paquete, quiet=True)


_asegurar_recursos_nltk()


# --------------------------------------------------------------------------
# Negaciones: nunca se tratan como stopwords, en ninguno de los dos modos.
#
# El material de clase muestra que quitar "no" invierte la polaridad de una
# frase. El ejercicio de hoy es de tema, no de opinión, pero la lista se
# restringe igual para dejarlo consistente con la limpieza de sentimiento de
# la entrega final.
# --------------------------------------------------------------------------
NEGACIONES = {"no", "not", "never", "nothing", "neither", "nor", "n't", "without", "none"}

STOPWORDS_INGLES = set(nltk_stopwords.words("english")) - NEGACIONES

# Stopwords específicas de esta colección. No están en la lista estándar de
# NLTK. Se aplica este criterio sobre el corpus ya sin stopwords estándar:
# la palabra aparece en más del 1% de los tweets como frecuencia documental Y
# su log-ratio P(palabra | desastre) / P(palabra | no_desastre), con
# suavizado de Laplace, tiene valor absoluto menor a 0.25. Eso quiere decir
# que no distingue clase y es ruido de dominio. El detalle de este análisis
# está en notebooks/02_preprocesamiento.ipynb, sección "Stopwords de dominio".
#
# El análisis exploratorio de 02_eda.md señaló "people", "still" y "burning"
# como candidatas por frecuencia compartida cruda. Al medirlas con este
# criterio más estricto de frecuencia documental más log-ratio suavizado,
# "people" con log-ratio 0.40 y "burning" con 0.42 resultan demasiado
# discriminativas para descartarlas. Solo "still" se confirma. "death"
# también cumple el criterio numérico, log-ratio 0.22, pero se excluye por
# criterio de dominio: es vocabulario central de la tarea, no ruido.
STOPWORDS_DOMINIO = {"video", "still", "us", "man", "rt", "first", "world", "say", "could"}

STOPWORDS_CLASIFICACION = STOPWORDS_INGLES | STOPWORDS_DOMINIO


# --------------------------------------------------------------------------
# Patrones de limpieza
# --------------------------------------------------------------------------
# El corpus no tiene contenido Unicode legítimo: todo carácter fuera de ASCII
# que aparece en el texto crudo es mojibake, texto UTF-8 mal decodificado al
# construir el archivo original. Un barrido de todo el texto crudo confirma
# que esos caracteres caen en el bloque Latin-1 Suplementario más controles
# C1, 0x80 a 0xFF, además del carácter de reemplazo U+FFFD. No son
# recuperables: el texto original ya llegó corrupto, así que se eliminan en
# vez de intentar redecodificarlos.
RE_MOJIBAKE = re.compile(r"[\x80-\xff�]+")

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENCION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_ENTIDAD_HTML = re.compile(r"&(amp|gt|lt|quot|nbsp);")
RE_ESPACIOS = re.compile(r"\s+")

# Emoticones ASCII: cara con ojos [:;=], nariz opcional, boca. `<3` aparte.
RE_EMOTICON_FELIZ = re.compile(r"(?<![A-Za-z0-9])[:;=][-o*']?[)\]dD](?![0-9A-Za-z])")
RE_EMOTICON_TRISTE = re.compile(r"(?<![A-Za-z0-9])[:;=][-o*']?[(\[](?![0-9A-Za-z])")
RE_EMOTICON_AMOR = re.compile(r"(?<![A-Za-z0-9])<3")

STEMMER = SnowballStemmer("english")
LEMATIZADOR = WordNetLemmatizer()


def _arreglar_entidades_y_mojibake(texto: str) -> str:
    t = RE_MOJIBAKE.sub(" ", texto)
    t = html.unescape(t)
    t = RE_ENTIDAD_HTML.sub(" ", t)
    return t


def _traducir_emoticones(texto: str) -> str:
    """Solo para la limpieza de sentimiento. El emoticón carga opinión, así
    que se traduce a un token de palabra en vez de perderse."""
    t = RE_EMOTICON_AMOR.sub(" emoji_amor ", texto)
    t = RE_EMOTICON_FELIZ.sub(" emoji_feliz ", t)
    t = RE_EMOTICON_TRISTE.sub(" emoji_triste ", t)
    return t


def _quitar_emoticones(texto: str) -> str:
    """Solo para la limpieza de clasificación. El emoticón no aporta al tema."""
    t = RE_EMOTICON_AMOR.sub(" ", texto)
    t = RE_EMOTICON_FELIZ.sub(" ", t)
    t = RE_EMOTICON_TRISTE.sub(" ", t)
    return t


def tokenizar(texto: str) -> list:
    """Tokeniza con el tokenizador de Treebank de NLTK, `word_tokenize`.

    Se usa ese tokenizador y no una separación ingenua por espacios porque es
    el que separa las contracciones en dos tokens: `isn't` se convierte en
    `is` más `n't`, que es exactamente la forma en que `NEGACIONES` espera
    encontrar la partícula de negación. Se conserva `n't` aunque no sea
    alfanumérico, porque es la partícula de negación.

    El tokenizador de Treebank no separa un guion o dos puntos pegados a
    letras, así que un token crudo como `err:509` o `self-harm` llega entero
    y no es alfanumérico. En vez de descartar ese token completo, se extraen
    sus fragmentos alfanuméricos internos: `err:509` produce `err` y `509`.
    Sin este paso algunos tweets cortos quedaban completamente vacíos después
    de la limpieza.
    """
    crudos = nltk.word_tokenize(texto)
    tokens = []
    for t in crudos:
        if t == "n't" or t.isalnum():
            tokens.append(t)
        else:
            tokens.extend(re.findall(r"[A-Za-z0-9]+", t))
    return tokens


def limpiar_clasificacion(texto: str) -> str:
    """Limpieza agresiva para el ejercicio 6, la clasificación por tema.

    Pasos, en este orden, cada uno documentado con ejemplo en
    notebooks/02_preprocesamiento.ipynb:

    1. minúsculas
    2. arreglar mojibake y entidades HTML, `&amp;` se convierte en nada y no
       en el carácter "&"
    3. URLs se convierten en el token uniforme `urlweb`. No se borran porque
       el análisis exploratorio midió que su sola presencia es el patrón más
       discriminativo del corpus, 67.35% frente a 41.73%. Lo que es ruido es
       el texto de la URL, no el hecho de que exista.
    4. menciones `@usuario` se borran
    5. hashtags: se quita el símbolo, se conserva la palabra
    6. emoticones se borran, a diferencia de la limpieza de sentimiento
    7. tokenización, que separa contracciones y descarta puntuación suelta
    8. stopwords: lista de NLTK en inglés, sin negaciones, más las stopwords
       de dominio de esta colección

    Los números NO se quitan aquí. `quitar_numeros` es un parámetro de
    `pipeline_texto`, no de esta función, porque el análisis exploratorio
    midió que los dígitos también son señal discriminativa, 72.74% frente a
    49.90%, y la decisión de conservarlos o no debe poder tomarse por
    separado.
    """
    t = texto.lower()
    t = _arreglar_entidades_y_mojibake(t)
    t = RE_URL.sub(" urlweb ", t)
    t = RE_MENCION.sub(" ", t)
    t = RE_HASHTAG.sub(r" \1 ", t)
    t = _quitar_emoticones(t)

    tokens = tokenizar(t)
    tokens = [tok for tok in tokens if tok not in STOPWORDS_CLASIFICACION]
    return " ".join(tokens)


def limpiar_sentimiento(texto: str) -> str:
    """Limpieza conservadora para los ejercicios 8-10 (entrega final).

    Conserva deliberadamente lo que `limpiar_clasificacion` descarta, porque
    en análisis de sentimientos es señal y no ruido:

    - mayúsculas, porque gritar en Twitter es señal de intensidad
    - signos de exclamación e interrogación
    - negaciones, que nunca se filtran; aquí ni siquiera se quitan stopwords
    - emoticones: se TRADUCEN a un token de palabra, `emoji_feliz`,
      `emoji_triste` o `emoji_amor`, en vez de borrarse, porque el material
      de clase señala que cargan opinión

    Sí se eliminan las URLs y las menciones completas. A diferencia de la
    limpieza de clasificación, aquí no importa que la URL exista: no aporta
    opinión. También se arregla el mojibake, que es corrupción del archivo y
    no señal de ningún tipo.
    """
    t = _arreglar_entidades_y_mojibake(texto)
    t = RE_URL.sub(" ", t)
    t = RE_MENCION.sub(" ", t)
    t = RE_HASHTAG.sub(r" \1 ", t)
    t = _traducir_emoticones(t)
    t = RE_ESPACIOS.sub(" ", t).strip()
    return t


def _reducir(tokens: list, reduccion: str) -> list:
    if reduccion == "ninguna":
        return tokens
    if reduccion == "stem":
        return [STEMMER.stem(t) for t in tokens]
    if reduccion == "lema":
        return [LEMATIZADOR.lemmatize(t) for t in tokens]
    raise ValueError(f"reduccion debe ser 'lema', 'stem' o 'ninguna'; recibido {reduccion!r}")


def pipeline_texto(
    texto: str,
    modo: str = "clasificacion",
    reduccion: str = "lema",
    quitar_numeros: bool = True,
) -> list:
    """Limpia, tokeniza y reduce un tweet. Es la función que usan los
    notebooks para poblar `text_clean` / `text_stem` / `text_sent`.

    Args:
        texto: tweet crudo.
        modo: 'clasificacion' aplica `limpiar_clasificacion` y quita
            stopwords. 'sentimiento' aplica `limpiar_sentimiento` y NO quita
            stopwords, porque la sintaxis completa importa para medir opinión.
        reduccion: 'lema' usa WordNetLemmatizer, 'stem' usa SnowballStemmer,
            'ninguna' deja los tokens sin reducir.
        quitar_numeros: si True, descarta los tokens que son puramente
            numéricos, por ejemplo `911` o `2015`, después de la limpieza.

    Returns:
        Lista de tokens.
    """
    if modo == "clasificacion":
        limpio = limpiar_clasificacion(texto)
        tokens = limpio.split()
    elif modo == "sentimiento":
        limpio = limpiar_sentimiento(texto)
        tokens = tokenizar(limpio)
    else:
        raise ValueError(f"modo debe ser 'clasificacion' o 'sentimiento'; recibido {modo!r}")

    if quitar_numeros:
        tokens = [t for t in tokens if not t.isdigit()]

    return _reducir(tokens, reduccion)


if __name__ == "__main__":
    ejemplos = [
        "Just happened a terrible car crash #Bridgetown http://t.co/abc123",
        "This isn't a drill!!! #earthquake @user 911 people missing",
        "my life is on fire lol :) love this song <3",
    ]
    for e in ejemplos:
        print("crudo:      ", e)
        print("clasific.:  ", limpiar_clasificacion(e))
        print("sentimiento:", limpiar_sentimiento(e))
        print("pipeline:   ", pipeline_texto(e))
        print()
