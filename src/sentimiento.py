"""Análisis de sentimiento del corpus de tweets: VADER, TextBlob y negatividad.

Este módulo es el único punto de entrada al análisis de sentimiento del laboratorio.
El reentrenamiento del ejercicio 10 depende de sus salidas, así que las firmas son un
contrato cerrado que no cambia:

    puntuar_vader(texto) -> dict
    etiquetar_polaridad(compound) -> str
    negatividad(compound) -> float
    anotar_dataframe(df, columna='text_sent') -> pd.DataFrame

Más funciones auxiliares para el análisis a nivel de palabra del ejercicio 8, que no
forman parte del contrato pero sí del análisis: `lexico_del_corpus` y `contar_palabras`.


Por qué VADER
-------------
El corpus es en inglés y VADER (*Valence Aware Dictionary and sEntiment Reasoner*) está
calibrado precisamente sobre texto de redes sociales. A diferencia de un conteo simple
de palabras del léxico, maneja de forma nativa cuatro cosas que abundan en Twitter y que
cambian la polaridad de una frase:

- **negación**: "not bad" no es lo mismo que "bad"
- **intensificadores**: "very good" pesa más que "good"
- **MAYÚSCULAS**: gritar es señal de intensidad
- **signos de exclamación** y **emoticones**

Es un enfoque basado en léxico, que no requiere datos etiquetados de sentimiento. Este
conjunto no los tiene: su única etiqueta es `target`, que indica si el tweet habla de un
desastre real, no si es positivo o negativo. Entrenar un clasificador supervisado de
sentimiento aquí es imposible, y por eso el enfoque léxico no es una comodidad sino la
única vía disponible.

Referencia: Hutto, C.J. & Gilbert, E.E. (2014). *VADER: A Parsimonious Rule-based Model
for Sentiment Analysis of Social Media Text*. Eighth International Conference on Weblogs
and Social Media (ICWSM-14).


Sobre qué texto se puntúa
-------------------------
Es la decisión fina de todo el ejercicio 8 y va contra lo que sugiere la intuición.

`text_clean` ya perdió las mayúsculas, la puntuación, los emoticones y las stopwords.
Alimentar eso a VADER destruye exactamente la señal que VADER sabe leer: sin `not` no hay
negación que detectar, y sin `!` no hay intensidad que medir. Por eso el valor por defecto
de `anotar_dataframe` es `text_sent`, la limpieza conservadora de `preprocesamiento.py`,
que quita URLs y menciones —ruido que no carga opinión— y conserva todo lo demás.

El cuaderno `07_sentimiento.ipynb` compara las tres versiones del texto y mide cuántos
tweets cambian de etiqueta según cuál se use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# --------------------------------------------------------------------------
# Recursos de NLTK: se descargan una sola vez, de forma perezosa e idempotente.
# Mismo patrón que usa `preprocesamiento.py`.
# --------------------------------------------------------------------------
def _asegurar_lexico_vader() -> None:
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


_ANALIZADOR = None


def _analizador():
    """Devuelve el analizador de VADER, construyéndolo una sola vez.

    Construir un `SentimentIntensityAnalyzer` implica leer y parsear un léxico de más de
    7,500 entradas. Hacerlo dentro de un `.apply` sobre 7,485 tweets lo repetiría una vez
    por fila. Con este singleton se paga una sola vez.
    """
    global _ANALIZADOR
    if _ANALIZADOR is None:
        _asegurar_lexico_vader()
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        _ANALIZADOR = SentimentIntensityAnalyzer()
    return _ANALIZADOR


# --------------------------------------------------------------------------
# Funciones del contrato
# --------------------------------------------------------------------------
def puntuar_vader(texto: str) -> dict:
    """Puntúa un texto con VADER.

    Args:
        texto: el texto a puntuar. Se recomienda pasar `text_sent` y no `text_clean`,
            por el motivo explicado en el encabezado del módulo.

    Returns:
        Diccionario con cuatro claves:
        - `neg`, `neu`, `pos`: proporción del texto que cae en cada categoría. Suman 1.
        - `compound`: puntuación agregada y normalizada en el rango [-1, 1]. Es la que se
          usa para etiquetar, porque es la única que integra negación e intensificadores
          en un solo número.

        Un texto vacío o nulo devuelve el resultado neutro `compound = 0.0`, que es lo
        que VADER produce para una cadena vacía. No se lanza excepción: un tweet que
        queda vacío tras limpiar es un caso real del corpus y debe poder puntuarse.
    """
    if texto is None or (isinstance(texto, float) and pd.isna(texto)):
        texto = ""
    return _analizador().polarity_scores(str(texto))


def etiquetar_polaridad(compound: float) -> str:
    """Traduce la puntuación `compound` a una etiqueta de tres categorías.

    Los umbrales son los estándar de VADER y viven en `src/config.py`:
    `compound >= 0.05` es positivo, `compound <= -0.05` es negativo, y el intervalo
    intermedio es neutro.

    Son una decisión del equipo, no una ley, y por eso están declarados como constantes
    y no incrustados aquí: el cuaderno analiza qué pasa al moverlos a 0.1 y a 0.2.

    Returns:
        'positivo', 'negativo' o 'neutro'.
    """
    if compound >= config.UMBRAL_POSITIVO:
        return "positivo"
    if compound <= config.UMBRAL_NEGATIVO:
        return "negativo"
    return "neutro"


def negatividad(compound: float) -> float:
    """Convierte `compound` en una medida de negatividad en el rango [0, 1].

        negatividad = (1 - compound) / 2

    Es una transformación lineal del `compound`, que vive en [-1, 1]. Un tweet
    máximamente positivo (`compound = 1`) da negatividad 0; uno máximamente negativo
    (`compound = -1`) da 1; uno neutro da 0.5.

    Esta es la variable que el ejercicio 10 incorpora al modelo de clasificación. Se
    define en [0, 1] a propósito: es el mismo rango en el que TF-IDF produce sus valores,
    así que puede concatenarse a la matriz de características sin escalar y sin que su
    magnitud domine artificialmente sobre la de los términos.

    Nótese que al ser una transformación monótona del `compound`, para un modelo lineal
    aporta exactamente la misma información que el `compound` mismo. La diferencia es de
    interpretación, no de contenido: "negatividad" se lee mejor en el informe.
    """
    return (1.0 - compound) / 2.0


def anotar_dataframe(df: pd.DataFrame, columna: str = "text_sent") -> pd.DataFrame:
    """Añade al DataFrame todas las columnas de sentimiento del laboratorio.

    Args:
        df: DataFrame con, al menos, la columna indicada en `columna`.
        columna: columna de texto sobre la que se puntúa. Por defecto `text_sent`, la
            limpieza conservadora. Pasar `text` para puntuar el crudo o `text_clean`
            para medir el efecto de la limpieza agresiva.

    Returns:
        Una **copia** del DataFrame con ocho columnas nuevas. No se modifica el original,
        para que el cuaderno pueda anotar el mismo DataFrame con tres columnas de texto
        distintas y comparar los resultados.

        | Columna | Contenido |
        |---|---|
        | `vader_neg`, `vader_neu`, `vader_pos` | proporciones de VADER, suman 1 |
        | `vader_compound` | puntuación agregada en [-1, 1] |
        | `polaridad` | 'positivo' / 'negativo' / 'neutro' |
        | `negatividad` | (1 - compound) / 2, en [0, 1] |
        | `textblob_polarity` | polaridad de TextBlob en [-1, 1], para contraste |
        | `textblob_subjectivity` | subjetividad de TextBlob en [0, 1] |

    Raises:
        KeyError: si `columna` no está en el DataFrame.
    """
    if columna not in df.columns:
        raise KeyError(
            f"El DataFrame no tiene la columna {columna!r}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    salida = df.copy()
    puntuaciones = salida[columna].apply(puntuar_vader)

    salida["vader_neg"] = puntuaciones.map(lambda d: d["neg"])
    salida["vader_neu"] = puntuaciones.map(lambda d: d["neu"])
    salida["vader_pos"] = puntuaciones.map(lambda d: d["pos"])
    salida["vader_compound"] = puntuaciones.map(lambda d: d["compound"])

    salida["polaridad"] = salida["vader_compound"].map(etiquetar_polaridad)
    salida["negatividad"] = salida["vader_compound"].map(negatividad)

    tb = salida[columna].apply(_puntuar_textblob)
    salida["textblob_polarity"] = tb.map(lambda t: t[0])
    salida["textblob_subjectivity"] = tb.map(lambda t: t[1])

    return salida


# --------------------------------------------------------------------------
# Texto optimizado para VADER
# --------------------------------------------------------------------------
def texto_para_vader(texto: str) -> str:
    """Limpieza conservadora que deja los emoticones **literales**, para puntuar con VADER.

    Es una variante de `limpiar_sentimiento` de `preprocesamiento.py` y existe por un
    motivo medido, no por preferencia.

    `limpiar_sentimiento` **traduce** los emoticones a tokens de palabra: `:)` se
    convierte en `emoji_feliz`. Esa decisión es correcta para un clasificador de bolsa de
    palabras, donde un tokenizador descartaría `:)` como puntuación suelta y la señal se
    perdería. Pero es contraproducente para VADER, porque **VADER ya trae los emoticones
    en su léxico**: `:)` vale 2.0, `:D` vale 2.3, `:(` vale -1.9 y `<3` vale 1.9. En
    cambio `emoji_feliz` no existe en el léxico y VADER lo ignora por completo.

    El efecto está medido en `notebooks/07_sentimiento.ipynb`: de los 52 tweets del
    corpus con emoticón ASCII, **19 cambian de etiqueta de polaridad** al traducirlos.

    Hace todo lo que hace `limpiar_sentimiento` —corregir mojibake y entidades HTML,
    eliminar URLs y menciones, quitar el símbolo del hashtag conservando la palabra— y
    conserva mayúsculas, puntuación expresiva, negaciones y emoticones tal cual.
    """
    from preprocesamiento import (RE_ENTIDAD_HTML, RE_ESPACIOS, RE_HASHTAG,
                                  RE_MENCION, RE_MOJIBAKE, RE_URL)
    import html as _html

    if texto is None or (isinstance(texto, float) and pd.isna(texto)):
        return ""

    t = RE_MOJIBAKE.sub(" ", str(texto))
    t = _html.unescape(t)
    t = RE_ENTIDAD_HTML.sub(" ", t)
    t = RE_URL.sub(" ", t)
    t = RE_MENCION.sub(" ", t)
    t = RE_HASHTAG.sub(r" \1 ", t)
    return RE_ESPACIOS.sub(" ", t).strip()


# --------------------------------------------------------------------------
# TextBlob: la segunda opinión
# --------------------------------------------------------------------------
def _puntuar_textblob(texto: str) -> tuple:
    """Devuelve (polaridad, subjetividad) de TextBlob, o (0.0, 0.0) si no está instalado.

    TextBlob es un contraste deliberado, no la herramienta principal. Usa el léxico de
    *pattern*, pensado para prosa general y no para redes sociales, y no maneja
    emoticones. Que discrepe de VADER no es un fallo de ninguna de las dos: es la
    evidencia de que medir sentimiento depende del método, y así se reporta.

    Si el paquete no está disponible el análisis principal no debe caerse, porque el
    contraste es opcional. Se devuelve un valor neutro y el cuaderno lo detecta.
    """
    if texto is None or (isinstance(texto, float) and pd.isna(texto)):
        return (0.0, 0.0)
    try:
        from textblob import TextBlob
    except ImportError:
        return (0.0, 0.0)
    sentimiento = TextBlob(str(texto)).sentiment
    return (sentimiento.polarity, sentimiento.subjectivity)


def textblob_disponible() -> bool:
    """Indica si TextBlob se puede importar, para que el cuaderno decida si contrastar."""
    try:
        import textblob  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------
# Análisis a nivel de palabra (ejercicio 8, primera mitad)
# --------------------------------------------------------------------------
def lexico_del_corpus(vocabulario) -> pd.DataFrame:
    """Cruza el vocabulario del corpus con el léxico de VADER.

    El enunciado pide *"determinar las palabras positivas, negativas o neutras"*. El
    léxico de VADER trae más de 7,500 términos puntuados a mano por anotadores humanos,
    pero listarlos todos no dice nada sobre **este** corpus. Lo que interesa es la
    intersección: qué palabras del léxico aparecen efectivamente en los tweets.

    Args:
        vocabulario: iterable de palabras (por ejemplo las claves de un `Counter`).

    Returns:
        DataFrame con columnas `palabra`, `valencia` y `polaridad`, ordenado de la más
        negativa a la más positiva. La valencia del léxico de VADER va de -4 a +4.
    """
    lex = _analizador().lexicon
    filas = [
        {"palabra": p, "valencia": lex[p]}
        for p in set(vocabulario)
        if p in lex
    ]
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return pd.DataFrame(columns=["palabra", "valencia", "polaridad"])
    tabla["polaridad"] = tabla["valencia"].map(
        lambda v: "positiva" if v > 0 else ("negativa" if v < 0 else "neutra"))
    return tabla.sort_values("valencia").reset_index(drop=True)


def contar_palabras(tokens) -> dict:
    """Cuenta cuántos tokens de un tweet son positivos, negativos o neutros.

    Implementa de forma literal lo que pide el enunciado: *"teniendo en cuenta la
    cantidad de palabras positivas y negativas del tweet determine qué tan positivo,
    negativo o neutral es el mismo"*.

    Es un conteo deliberadamente ingenuo, y esa es su utilidad. **No** maneja negación:
    en "not bad", `bad` cuenta como una palabra negativa y `not` no cuenta como nada, así
    que el conteo declara el tweet negativo cuando en realidad es levemente positivo.
    Contrastar este conteo con el `compound` de VADER es lo que demuestra en el cuaderno
    por qué hace falta un modelo con reglas y no basta con contar.

    Args:
        tokens: lista de tokens del tweet.

    Returns:
        `{'n_positivas': int, 'n_negativas': int, 'n_neutras': int, 'balance': int}`,
        donde `balance` es positivas menos negativas.
    """
    lex = _analizador().lexicon
    n_pos = n_neg = n_neu = 0
    for t in tokens:
        v = lex.get(t.lower())
        if v is None or v == 0:
            n_neu += 1
        elif v > 0:
            n_pos += 1
        else:
            n_neg += 1
    return {
        "n_positivas": n_pos,
        "n_negativas": n_neg,
        "n_neutras": n_neu,
        "balance": n_pos - n_neg,
    }


if __name__ == "__main__":
    ejemplos = [
        "Forest fire near La Ronge Sask. Canada",
        "This is not bad at all!",
        "I LOVE this song so much :)",
        "13,000 people receive #wildfires evacuation orders in California",
        "",
    ]
    print(f"{'texto':<62} {'compound':>9} {'polaridad':>10} {'negatividad':>12}")
    print("-" * 96)
    for e in ejemplos:
        p = puntuar_vader(e)
        print(f"{(e or '(vacio)'):<62} {p['compound']:>9.4f} "
              f"{etiquetar_polaridad(p['compound']):>10} {negatividad(p['compound']):>12.4f}")
