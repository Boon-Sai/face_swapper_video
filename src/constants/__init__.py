import os

# ==PATHS==

CURRENT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR_PATH, "../../"))

# Input data folder
DATA_DIR_PATH = os.path.join(PROJECT_ROOT, "data")

# Main LOGS folder path
LOG_DIR_PATH = os.path.join(PROJECT_ROOT, "logs")

# Main ARTIFACT folder path 
ARTIFACT_DIR_PATH = os.path.join(PROJECT_ROOT, "artifacts")

# ==MODELS==

DETECTION_MODEL = "buffalo_l"
SWAPPING_MODEL = "weights/inswapper_128.onnx"
CTX_ID = 0
DET_SIZE = (640, 640)

DIRECT_URL = "https://drive.usercontent.google.com/download?id=1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF&export=download&authuser=0"


CLUSTERING_MODEL = "DBSCAN"
AUDIO_MODEL = "ffmgep"