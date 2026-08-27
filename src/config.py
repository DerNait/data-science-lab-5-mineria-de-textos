"""Configuración central del Laboratorio 5.

Todas las constantes del laboratorio viven aquí. Ningún notebook define su propia
semilla, su propia ruta ni su propio umbral: los importa de este módulo. Así los
resultados de las tres personas del equipo son comparables entre sí.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
MODELOS = RESULTS / "modelos"

INFORME = ROOT / "informe"
SECCIONES = INFORME / "secciones"

TRAIN_CSV = RAW / "train.csv"

# Partición congelada: la produce Persona C y la consumen todos.
# Nadie vuelve a partir los datos por su cuenta (ver plan maestro, sección 5.2).
PARTICION = PROCESSED / "particion.parquet"

# DataFrame anotado con sentimiento: lo produce Persona A en la entrega final.
TWEETS_SENTIMIENTO = PROCESSED / "tweets_con_sentimiento.parquet"


# --------------------------------------------------------------------------
# Reproducibilidad
# --------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20


# --------------------------------------------------------------------------
# Etiquetas del problema
# --------------------------------------------------------------------------
# target del dataset: 1 = el tweet se refiere a un desastre real, 0 = no.
ETIQUETAS = {0: "no desastre", 1: "desastre"}
CLASE_POSITIVA = 1  # la clase de interés: no detectar un desastre real es el error caro


# --------------------------------------------------------------------------
# Umbrales de polaridad de VADER (entrega final, ejercicios 8 a 10)
# --------------------------------------------------------------------------
# Son los umbrales estándar de VADER. Se declaran aquí porque son una decisión
# del equipo y hay que poder discutir su sensibilidad en el informe.
UMBRAL_POSITIVO = 0.05
UMBRAL_NEGATIVO = -0.05


# --------------------------------------------------------------------------
# Presentación de figuras
# --------------------------------------------------------------------------
FIG_DPI = 150
FIG_ANCHO = 10
FIG_ALTO = 5

# Paleta consistente por clase en todas las gráficas del laboratorio.
COLOR_DESASTRE = "#c0392b"      # rojo: desastre real
COLOR_NO_DESASTRE = "#2980b9"   # azul: no desastre
COLOR_NEUTRO = "#7f8c8d"

COLORES_CLASE = {0: COLOR_NO_DESASTRE, 1: COLOR_DESASTRE}


def asegurar_directorios() -> None:
    """Crea los directorios de salida si no existen."""
    for d in (RAW, PROCESSED, FIGURES, TABLES, MODELOS, SECCIONES):
        d.mkdir(parents=True, exist_ok=True)
