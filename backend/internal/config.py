import os
import xgboost as xgb
MODEL_PATH = "coal_fire_model.json"
ml_model = None

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