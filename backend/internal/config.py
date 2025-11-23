import joblib
from pathlib import Path
import os

# В Docker контейнере WORKDIR /app, поэтому пути строим оттуда
BASE_DIR = Path("/app")
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODELS_DIR / "coal_fire_model.pkl"
ENCODER_PATH = MODELS_DIR / "coal_label_encoder.pkl"

# Глобальные переменные
ml_model = None
label_encoder = None

def loadConfig():
    global ml_model, label_encoder

    # Создаем папку данных, если нет
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Загрузка модели
    if MODEL_PATH.exists():
        try:
            ml_model = joblib.load(MODEL_PATH)
            print(f"✅ ML Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Failed to load ML Model: {e}")
    else:
        print(f"⚠️ Model file not found: {MODEL_PATH}")

    # Загрузка энкодера
    if ENCODER_PATH.exists():
        try:
            label_encoder = joblib.load(ENCODER_PATH)
            print(f"✅ Label Encoder loaded from {ENCODER_PATH}")
        except Exception as e:
            print(f"❌ Failed to load Label Encoder: {e}")
    else:
        print(f"⚠️ Encoder file not found: {ENCODER_PATH}")