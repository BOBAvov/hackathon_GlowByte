import os
import xgboost as xgb

from pathlib import Path

MODEL_PATH = "coal_fire_model.json"

ml_model = None

DATA_DIR = Path('/app/data')

def loadConfig():
    if os.path.exists(MODEL_PATH):
        try:
            ml_model = xgb.XGBClassifier()
            print(os.path)
            ml_model.load_model(MODEL_PATH)
            print(f"✅ Модель успешно загружена из {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
    else:
        print(f"⚠️ Файл модели {MODEL_PATH} не найден!")