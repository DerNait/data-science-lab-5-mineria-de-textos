"""Obtiene `train.csv` del dataset Disaster Tweets y lo deja en data/raw/.

El dataset pertenece a la competencia de Kaggle *Natural Language Processing with
Disaster Tweets* (`nlp-getting-started`). Al ser una competencia y no un dataset
público, `kagglehub` exige credenciales y haber aceptado las reglas desde la web.
Por eso el script intenta la vía oficial primero y, si no hay credenciales,
explica con precisión qué hacer a mano en lugar de fallar en silencio.

Uso:
    python src/download_dataset.py
    python src/download_dataset.py --verificar    # solo valida lo ya descargado
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

COMPETENCIA = "nlp-getting-started"
URL_KAGGLE = "https://www.kaggle.com/competitions/nlp-getting-started/data"

# Huella del archivo verificado el 27 de agosto de 2026. Sirve para detectar que
# alguien bajó un CSV distinto (por ejemplo el `test.csv`, que no trae `target`).
SHA256_ESPERADO = "61111c6dc31eaffa34d1e1fa62e2395325c9bc3b38bba1941a5f1ed9b3fa60df"
FILAS_ESPERADAS = 7613
COLUMNAS_ESPERADAS = ["id", "keyword", "location", "text", "target"]


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def verificar(ruta: Path) -> bool:
    """Comprueba que el CSV en disco sea el esperado. Devuelve True si todo cuadra."""
    if not ruta.exists():
        print(f"[x] No existe {ruta}")
        return False

    import pandas as pd

    df = pd.read_csv(ruta)
    ok = True

    huella = sha256(ruta)
    if huella == SHA256_ESPERADO:
        print(f"[ok] SHA256 coincide con el archivo de referencia.")
    else:
        print(f"[!] SHA256 distinto al de referencia.")
        print(f"    esperado: {SHA256_ESPERADO}")
        print(f"    obtenido: {huella}")
        print("    No es necesariamente un error, pero verifica que no sea test.csv.")

    faltantes = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    if faltantes:
        print(f"[x] Faltan columnas: {faltantes}")
        if "target" in faltantes:
            print("    Parece que descargaste `test.csv`. Ese archivo no trae etiquetas:")
            print("    es el conjunto ciego de la competencia. Necesitas `train.csv`.")
        ok = False
    else:
        print(f"[ok] Las 5 columnas esperadas están presentes.")

    if len(df) != FILAS_ESPERADAS:
        print(f"[!] Se esperaban {FILAS_ESPERADAS:,} filas y hay {len(df):,}.")
    else:
        print(f"[ok] {len(df):,} filas, como se esperaba.")

    if "target" in df.columns:
        valores = sorted(df["target"].dropna().unique().tolist())
        if valores != [0, 1]:
            print(f"[x] `target` debería contener solo 0 y 1, contiene {valores}")
            ok = False
        else:
            conteo = df["target"].value_counts().sort_index()
            print(f"[ok] target: {conteo[0]:,} no-desastre / {conteo[1]:,} desastre")

    return ok


def descargar_con_kagglehub(destino: Path) -> bool:
    """Intenta la descarga oficial. Devuelve True si lo logró."""
    try:
        import kagglehub
    except ImportError:
        print("[!] `kagglehub` no está instalado (pip install -r requirements.txt).")
        return False

    try:
        print(f"Descargando la competencia {COMPETENCIA} desde Kaggle...")
        carpeta = Path(kagglehub.competition_download(COMPETENCIA))
    except Exception as exc:  # credenciales ausentes, reglas no aceptadas, red
        print(f"[!] La descarga automática falló: {type(exc).__name__}: {exc}")
        return False

    candidatos = sorted(carpeta.rglob("train.csv"))
    if not candidatos:
        print(f"[!] No se encontró train.csv dentro de {carpeta}")
        return False

    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidatos[0], destino)
    print(f"[ok] Guardado en {destino}")
    return True


def instrucciones_manuales(destino: Path) -> None:
    print()
    print("=" * 74)
    print("DESCARGA MANUAL")
    print("=" * 74)
    print("La descarga automática requiere credenciales de Kaggle y haber aceptado")
    print("las reglas de la competencia desde el navegador. Para hacerlo a mano:")
    print()
    print(f"  1. Abre {URL_KAGGLE}")
    print("  2. Inicia sesión y pulsa 'Join Competition' para aceptar las reglas.")
    print("  3. Descarga `train.csv`.")
    print(f"  4. Cópialo a: {destino}")
    print()
    print("Para habilitar la vía automática en el futuro, coloca tu token de la API")
    print("de Kaggle en ~/.kaggle/kaggle.json (Account > Create New API Token).")
    print("=" * 74)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verificar", action="store_true",
                        help="solo valida el archivo ya presente, sin descargar")
    args = parser.parse_args()

    destino = config.TRAIN_CSV

    if args.verificar:
        sys.exit(0 if verificar(destino) else 1)

    if destino.exists():
        print(f"Ya existe: {destino}")
        print("Verificando...")
        sys.exit(0 if verificar(destino) else 1)

    if descargar_con_kagglehub(destino):
        sys.exit(0 if verificar(destino) else 1)

    instrucciones_manuales(destino)
    sys.exit(1)


if __name__ == "__main__":
    main()
