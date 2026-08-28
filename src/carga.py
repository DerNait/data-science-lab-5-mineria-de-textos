"""Carga, validación y deduplicación del corpus de tweets.

Este módulo es el único punto de entrada a los datos. Todos los notebooks del
laboratorio llaman a `cargar_tweets()`; nadie lee el CSV directamente. Así la
regla de deduplicación se aplica una sola vez, de la misma forma, para todos.

Contrato de la función:

    cargar_tweets(deduplicar=True) -> pd.DataFrame

    con columnas garantizadas:
        id          int     identificador del tweet
        keyword     str     palabra clave, puede ser NaN
        location    str     ubicación autodeclarada, puede ser NaN
        text        str     texto crudo del tweet, SIN tocar
        target      int     1 = desastre real, 0 = no
        n_palabras  int     conteo de palabras del texto crudo
        n_caracteres int    longitud del texto crudo
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

COLUMNAS_ORIGINALES = ["id", "keyword", "location", "text", "target"]


class ErrorDeCarga(Exception):
    """El archivo de datos no cumple el contrato esperado."""


def _validar(df: pd.DataFrame) -> None:
    """Falla temprano y con un mensaje útil si el CSV no es el que creemos."""
    faltantes = [c for c in COLUMNAS_ORIGINALES if c not in df.columns]
    if faltantes:
        if "target" in faltantes:
            raise ErrorDeCarga(
                "El archivo no tiene la columna `target`. Es muy probable que sea "
                "`test.csv`, el conjunto ciego de la competencia, que no trae "
                "etiquetas. Se necesita `train.csv`."
            )
        raise ErrorDeCarga(f"Faltan columnas en el CSV: {faltantes}")

    valores = sorted(df["target"].dropna().unique().tolist())
    if valores != [0, 1]:
        raise ErrorDeCarga(f"`target` debería contener solo 0 y 1; contiene {valores}")

    if df["text"].isna().any():
        raise ErrorDeCarga("Hay tweets con `text` nulo, algo que no se esperaba.")


def _deduplicar(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Aplica la regla de deduplicación acordada por el equipo.

    Son dos problemas distintos y se tratan distinto:

    1. Textos repetidos con etiquetas CONTRADICTORIAS. El mismo tweet aparece
       marcado como desastre y como no-desastre. No hay forma de saber cuál es
       la correcta, así que se eliminan TODAS sus filas. Es ruido de etiqueta
       documentado del dataset y conservarlo solo mete confusión al modelo.

    2. Textos repetidos con etiqueta CONSISTENTE. Se conserva la primera
       aparición. Si no se hiciera, el mismo tweet podría caer en entrenamiento
       y en prueba a la vez, y la métrica saldría inflada.

    La deduplicación ocurre ANTES de partir en train/test. Ese orden importa.
    """
    n_inicial = len(df)

    # 1. Textos con etiquetas en conflicto
    etiquetas_por_texto = df.groupby("text")["target"].nunique()
    textos_en_conflicto = etiquetas_por_texto[etiquetas_por_texto > 1].index
    filas_en_conflicto = int(df["text"].isin(textos_en_conflicto).sum())
    df = df[~df["text"].isin(textos_en_conflicto)].copy()

    # 2. Duplicados exactos restantes
    n_antes_dup = len(df)
    df = df.drop_duplicates(subset="text", keep="first").copy()
    filas_duplicadas = n_antes_dup - len(df)

    if verbose:
        print("Deduplicación")
        print(f"  Filas originales................. {n_inicial:,}")
        print(f"  Textos con etiqueta contradictoria {len(textos_en_conflicto):,} "
              f"({filas_en_conflicto:,} filas eliminadas)")
        print(f"  Duplicados exactos eliminados.... {filas_duplicadas:,}")
        print(f"  Filas finales.................... {len(df):,} "
              f"({n_inicial - len(df):,} eliminadas en total)")

    return df


def cargar_tweets(deduplicar: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Lee `data/raw/train.csv`, lo valida y devuelve el DataFrame del laboratorio.

    Args:
        deduplicar: aplica la regla de deduplicación. Dejar en True salvo que se
            quiera analizar explícitamente el ruido de etiqueta del dataset.
        verbose: imprime el resumen de lo que se eliminó.

    Returns:
        DataFrame con las columnas del contrato.
    """
    if not config.TRAIN_CSV.exists():
        raise ErrorDeCarga(
            f"No se encontró {config.TRAIN_CSV}.\n"
            "Ejecuta primero:  python src/download_dataset.py"
        )

    df = pd.read_csv(config.TRAIN_CSV)
    _validar(df)

    if verbose:
        print(f"Leído {config.TRAIN_CSV.name}: {len(df):,} filas x {df.shape[1]} columnas")

    if deduplicar:
        df = _deduplicar(df, verbose=verbose)

    # Variables derivadas del texto CRUDO. Se calculan sobre el original a
    # propósito: describen el tweet tal como se escribió, no tras limpiarlo.
    df["n_palabras"] = df["text"].str.split().str.len()
    df["n_caracteres"] = df["text"].str.len()

    df["target"] = df["target"].astype(int)
    df = df.reset_index(drop=True)

    return df


def resumen(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de nulos y cardinalidad por columna, para el análisis exploratorio."""
    return pd.DataFrame({
        "tipo": df.dtypes.astype(str),
        "nulos": df.isna().sum(),
        "% nulos": (df.isna().mean() * 100).round(2),
        "valores_unicos": df.nunique(),
    })


if __name__ == "__main__":
    datos = cargar_tweets()
    print()
    print(resumen(datos))
    print()
    print("Distribución de target:")
    print(datos["target"].value_counts().rename(config.ETIQUETAS))
