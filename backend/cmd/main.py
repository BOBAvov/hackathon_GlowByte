import os
import io
import pandas as pd
import numpy as np
import xgboost as xgb
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Глобальные переменные
MODEL_PATH = "coal_fire_model.json"
bst_model = None


# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global bst_model
    print("🚀 Запуск приложения...")

    if os.path.exists(MODEL_PATH):
        try:
            bst_model = xgb.XGBClassifier()
            bst_model.load_model(MODEL_PATH)
            print(f"✅ Модель успешно загружена из {MODEL_PATH}")
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
    else:
        print(f"⚠️ Файл модели {MODEL_PATH} не найден!")

    yield
    print("🛑 Остановка приложения...")


app = FastAPI(title="Coal Fire Prediction API", version="2.3", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- ЛОГИКА ОБРАБОТКИ (ETL) ---

def process_data_pipeline(weather_content, supplies_content, temperature_content):
    try:
        # 1. Чтение
        weather = pd.read_csv(io.BytesIO(weather_content))
        supplies = pd.read_csv(io.BytesIO(supplies_content))
        temperature = pd.read_csv(io.BytesIO(temperature_content))

        # 2. Предобработка
        weather["date"] = pd.to_datetime(weather["date"], errors='coerce')
        weather["date_norm"] = weather["date"].dt.normalize()

        supplies["start_date"] = pd.to_datetime(supplies["ВыгрузкаНаСклад"], errors='coerce')
        supplies["end_date"] = pd.to_datetime(supplies["ПогрузкаНаСудно"], errors='coerce')

        temperature["date"] = pd.to_datetime(temperature["Дата акта"], errors='coerce')
        if 'Марка' in temperature.columns:
            temperature['Марка'] = temperature['Марка'].astype(str).apply(lambda x: x.split('-')[0])

        # 3. Разворачивание Supplies
        grouped_supplies = supplies.groupby(['Склад', 'Штабель']).agg(
            start_min=('start_date', 'min'),
            end_max=('end_date', 'max')
        ).reset_index()

        res_supplies = pd.DataFrame()

        for i, row in grouped_supplies.iterrows():
            if pd.isna(row['start_min']) or pd.isna(row['end_max']):
                continue

            dates_range = pd.date_range(start=row['start_min'], end=row['end_max'], freq='D')
            subset_supplies = supplies[(supplies['Склад'] == row['Склад']) & (supplies['Штабель'] == row['Штабель'])]
            unique_types = subset_supplies['Наим. ЕТСНГ'].unique()

            for type_col in unique_types:
                local_df = pd.DataFrame({'date': dates_range})
                local_df['Склад'] = row['Склад']
                local_df['Штабель'] = row['Штабель']
                local_df['Наим. ЕТСНГ'] = type_col

                temp = subset_supplies[subset_supplies['Наим. ЕТСНГ'] == type_col]
                if temp.empty:
                    local_df['Масса угля'] = 0.0
                else:
                    idx = pd.to_datetime(dates_range).normalize()
                    arrivals = temp.groupby(temp['start_date'].dt.normalize())['На склад, тн'].sum()
                    shipments = temp.groupby(temp['end_date'].dt.normalize())['На судно, тн'].sum()
                    stock = (arrivals.reindex(idx, fill_value=0).cumsum() -
                             shipments.reindex(idx, fill_value=0).cumsum()).clip(lower=0)
                    local_df['Масса угля'] = local_df['date'].dt.normalize().map(stock).fillna(0).astype(float)

                res_supplies = pd.concat([res_supplies, local_df], ignore_index=True)

        if res_supplies.empty:
            return {"status": "error", "message": "Empty supplies data"}

        # 4. Мердж данных
        res_supplies['date'] = pd.to_datetime(res_supplies['date'])
        res_supplies['date_norm'] = res_supplies['date'].dt.normalize()
        weather_agg = weather.groupby('date_norm').mean(numeric_only=True).reset_index()
        full_df = res_supplies.merge(weather_agg, on='date_norm', how='left')

        # Мердж с температурой
        temp_subset = temperature[['date', 'Склад', 'Штабель', 'Максимальная температура', 'Марка']].copy()
        full_df['Наим. ЕТСНГ'] = full_df['Наим. ЕТСНГ'].astype(str)
        temp_subset['Марка'] = temp_subset['Марка'].astype(str)

        full_df = full_df.merge(
            temp_subset,
            left_on=['date', 'Склад', 'Штабель', 'Наим. ЕТСНГ'],
            right_on=['date', 'Склад', 'Штабель', 'Марка'],
            how='left'
        )

        if 'Марка' in full_df.columns:
            full_df.drop(columns=['Марка'], inplace=True)

        # [FIX 1] Замена interpolate на ffill
        full_df['Максимальная температура'] = full_df.groupby(['Склад', 'Штабель'])[
            'Максимальная температура'].transform(lambda x: x.ffill().fillna(0))

        if 'visibility' in full_df.columns:
            full_df.drop(columns=['visibility'], inplace=True)

        # 5. Генерация признаков
        df_feat = full_df.sort_values(by=['Склад', 'Штабель', 'date'])
        base_features = ['t', 'humidity', 'wind_dir', 'v_avg', 'Масса угля', 'Максимальная температура']
        available_features = [f for f in base_features if f in df_feat.columns]

        for col in available_features:
            if col == 'Максимальная температура':
                continue
            for lag in [1, 3, 7]:
                df_feat[f'{col}_lag_{lag}'] = df_feat.groupby(['Склад', 'Штабель'])[col].shift(lag)

        for col in ['t', 'Масса угля']:
            if col in df_feat.columns:
                df_feat[f'{col}_ma_7'] = df_feat.groupby(['Склад', 'Штабель'])[col].transform(
                    lambda x: x.rolling(7).mean())


        numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
        df_feat[numeric_cols] = df_feat[numeric_cols].ffill().fillna(0)

        df_feat = df_feat[df_feat['Масса угля']>0]
        df_feat = df_feat.drop_duplicates(subset=['Склад','Штабель'], keep='last')


        # 6. Предсказание
        if bst_model and not df_feat.empty:
            booster = bst_model.get_booster()
            expected_features = booster.feature_names

            missing_cols = [c for c in expected_features if c not in df_feat.columns]
            if missing_cols:
                for c in missing_cols:
                    df_feat[c] = 0

            X_pred = df_feat[expected_features]
            probs = bst_model.predict_proba(X_pred)[:, 1]
            df_feat['fire_probability'] = probs
            df_feat['is_dangerous'] = (probs > 0.5).astype(int)
        else:
            df_feat['fire_probability'] = 0.0
            df_feat['is_dangerous'] = 0

        # 7. Результат
        df_feat = df_feat.replace([np.inf, -np.inf], 0).fillna(0)

        # [FIX 2] Добавили .copy() чтобы убрать Warning
        result_df = df_feat[['date', 'Склад', 'Штабель', 'Масса угля', 'fire_probability', 'is_dangerous']].copy()
        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')

        top_risks = result_df[result_df['fire_probability'] > 0.1].sort_values('fire_probability',
                                                                               ascending=False).head(200)

        return {
            "status": "success",
            "total_records": len(result_df),
            "high_risk_count": int(result_df['is_dangerous'].sum()),
            "top_risks": top_risks.to_dict(orient='records'),
            "warehouse_stats": result_df.groupby('Склад')['is_dangerous'].sum().to_dict()
        }

    except Exception as e:
        print(f"Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/predict-fire-risk")
async def predict_fire_risk(
        weather_file: UploadFile = File(...),
        supplies_file: UploadFile = File(...),
        temperature_file: UploadFile = File(...)
):
    w_content = await weather_file.read()
    s_content = await supplies_file.read()
    t_content = await temperature_file.read()

    result = process_data_pipeline(w_content, s_content, t_content)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)