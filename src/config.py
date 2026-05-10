import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(os.environ.get("SMARTSCHOOL_DATA", ROOT / "data" / "student_dataset")).resolve()
TRAIN_CSV = DATA_ROOT / "student_failure" / "train.csv"
IMAGE_DIR = DATA_ROOT / "image_data"

REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"
RES_DIR = REPORTS / "results"

TARGET = "score_examen"
ID_COL = "id"
RISK_THRESHOLD = 50.0  # consigne école : tutorat si note prédite < 50

# Variables retenues après EDA. Les autres sont dans NUM_COLS_DROPPED / CAT_COLS_DROPPED,
# je les garde listées pour pouvoir les remettre vite si besoin.
NUM_COLS_KEPT = ["heures_etude", "assiduité_classe", "heures_sommeil"]
NUM_COLS_DROPPED = ["age", "heures_fête", "taille_etudiant"]
NUM_COLS_ALL = NUM_COLS_KEPT + NUM_COLS_DROPPED

CAT_COLS_KEPT = [
    "genre",
    "diplôme",
    "accès_internet",
    "qualité_sommeil",
    "méthode_etude",
    "évaluation_établissement",
]
CAT_COLS_DROPPED = ["difficulté_examen"]
CAT_COLS_ALL = CAT_COLS_KEPT + CAT_COLS_DROPPED

DERIVED_COLS = ["study_efficiency", "sleep_score", "risk_internet_low"]

FEATURES_FINAL = NUM_COLS_KEPT + CAT_COLS_KEPT
FEATURES_FINAL_PLUS = NUM_COLS_KEPT + DERIVED_COLS + CAT_COLS_KEPT
FEATURES_ALL = NUM_COLS_ALL + CAT_COLS_ALL

# Pour le supplément Advanced AI (analyse d'équité)
SENSITIVE = "genre"
SENSITIVE_PROXY = ["diplôme"]

RANDOM_STATE = 42
