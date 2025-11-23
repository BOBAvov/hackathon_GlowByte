import io
import pandas as pd
import numpy as np
from internal import config


# --- FILE OPERATIONS ---

def get_saved_data(filename):
    path = config.DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"File {filename} not found")
    with open(path, 'rb') as f:
        return f.read()


def save_file(filename, data: bytes):
    path = config.DATA_DIR / filename
    with open(path, 'wb') as f:
        f.write(data)


# --- DATA PREPARATION PIPELINE ---

def prepare_raw_dataframe(weather_content, supplies_content, temperature_content):
    """
    Этап 1: Чтение, очистка и слияние сырых данных в единый временной ряд.
    """
    try:
        # 1. Чтение
        weather = pd.read_csv(io.BytesIO(weather_content))
        supplies = pd.read_csv(io.BytesIO(supplies_content))
        temperature = pd.read_csv(io.BytesIO(temperature_content))

        # 2. Типы данных
        weather['date'] = pd.to_datetime(weather['date'], errors='coerce')
        weather['date_norm'] = weather['date'].dt.normalize()

        supplies['start_date'] = pd.to_datetime(supplies['ВыгрузкаНаСклад'], errors='coerce')
        supplies['end_date'] = pd.to_datetime(supplies['ПогрузкаНаСудно'], errors='coerce')

        temperature['date'] = pd.to_datetime(temperature['Дата акта'], errors='coerce')
        if 'Марка' in temperature.columns:
            temperature['Марка'] = temperature['Марка'].astype(str).apply(lambda x: x.split('-')[0])

        # 3. Разворачивание Supplies (Supplies Explosion)
        # Создаем сетку дат для каждого штабеля
        grouped_supplies = supplies.groupby(['Склад', 'Штабель']).agg(
            start_min=('start_date', 'min'),
            end_max=('end_date', 'max')
        ).reset_index()

        res_dfs = []
        for _, row in grouped_supplies.iterrows():
            if pd.isna(row['start_min']) or pd.isna(row['end_max']):
                continue

            # Генерируем полный диапазон дат
            dates_range = pd.date_range(start=row['start_min'], end=row['end_max'], freq='D')

            # Фильтруем поставки для конкретного штабеля
            subset = supplies[(supplies['Склад'] == row['Склад']) & (supplies['Штабель'] == row['Штабель'])]

            for coal_type in subset['Наим. ЕТСНГ'].unique():
                local_df = pd.DataFrame({'date': dates_range})
                local_df['Склад'] = row['Склад']
                local_df['Штабель'] = row['Штабель']
                local_df['Наим. ЕТСНГ'] = coal_type

                # Расчет остатков
                temp_data = subset[subset['Наим. ЕТСНГ'] == coal_type]
                idx = pd.to_datetime(dates_range).normalize()

                arrivals = temp_data.groupby(temp_data['start_date'].dt.normalize())['На склад, тн'].sum()
                shipments = temp_data.groupby(temp_data['end_date'].dt.normalize())['На судно, тн'].sum()

                stock = (arrivals.reindex(idx, fill_value=0).cumsum() -
                         shipments.reindex(idx, fill_value=0).cumsum()).clip(lower=0)

                local_df['coal_mass'] = local_df['date'].dt.normalize().map(stock).fillna(0).astype(float)
                res_dfs.append(local_df)

        if not res_dfs:
            return {'status': 'error', 'message': 'No valid supply data found'}

        full_df = pd.concat(res_dfs, ignore_index=True)

        # 4. Объединение с погодой
        full_df['date'] = pd.to_datetime(full_df['date'])
        full_df['date_norm'] = full_df['date'].dt.normalize()

        weather_agg = weather.groupby('date_norm').mean(numeric_only=True).reset_index()
        full_df = full_df.merge(weather_agg, on='date_norm', how='left')

        # Заполняем пропуски погоды (ffill/bfill как при обучении)
        weather_cols = ['t', 'p', 'humidity', 'precipitation', 'wind_dir', 'v_avg', 'v_max', 'cloudcover']
        # Сначала убедимся, что колонки есть
        for c in weather_cols:
            if c not in full_df.columns: full_df[c] = 0

        full_df[weather_cols] = full_df.groupby(['Склад', 'Штабель'])[weather_cols].ffill().bfill()

        # 5. Объединение с температурой
        full_df['Наим. ЕТСНГ'] = full_df['Наим. ЕТСНГ'].astype(str)
        temperature['Марка'] = temperature['Марка'].astype(str)

        full_df = full_df.merge(
            temperature[['date', 'Склад', 'Штабель', 'Максимальная температура', 'Марка']],
            left_on=['date', 'Склад', 'Штабель', 'Наим. ЕТСНГ'],
            right_on=['date', 'Склад', 'Штабель', 'Марка'],
            how='left'
        )

        if 'Марка' in full_df.columns:
            del full_df['Марка']

        # Линейная интерполяция температуры (как при обучении)
        full_df['Максимальная температура'] = full_df.groupby(['Склад', 'Штабель', 'Наим. ЕТСНГ'])[
            'Максимальная температура'] \
            .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))

        # Если совсем нет данных, берем t воздуха
        full_df['Максимальная температура'] = full_df['Максимальная температура'].fillna(full_df['t'])

        # Чистка лишнего
        cols_to_drop = ['date_norm', 'visibility', 'weather_code']
        full_df.drop(columns=[c for c in cols_to_drop if c in full_df.columns], inplace=True)

        return full_df

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': f"Data Prep Error: {str(e)}"}


def feature_engineering(df):
    """
    Этап 2: Генерация признаков (Delta, Rolling), идентичных тем, на которых училась модель.
    """
    df = df.sort_values(['Склад', 'Штабель', 'Наим. ЕТСНГ', 'date'])
    g = df.groupby(['Склад', 'Штабель', 'Наим. ЕТСНГ'])

    # Те же колонки, что при обучении
    # ВАЖНО: Имена колонок в coal_mass переименованы в 'coal_mass' в prepare_raw_dataframe
    target_cols = ['coal_mass', 'Максимальная температура', 't', 'p', 'humidity',
                   'precipitation', 'wind_dir', 'v_avg', 'v_max', 'cloudcover']

    # Убедимся, что все колонки есть
    for c in target_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Генерация лагов и скользящих средних
    for col in target_cols:
        # Deltas
        df[f'{col}_delta_1d'] = g[col].diff(1).fillna(0)
        df[f'{col}_delta_3d'] = g[col].diff(3).fillna(0)

        # Rolling Stats (5 дней)
        df[f'{col}_mean_5d'] = g[col].transform(lambda x: x.rolling(5, min_periods=1).mean()).fillna(0)
        df[f'{col}_std_5d'] = g[col].transform(lambda x: x.rolling(5, min_periods=1).std()).fillna(0)

    return df


def process_data_pipeline(weather_content, supplies_content, temperature_content):
    """
    Главная функция пайплайна: Сырые данные -> Фичи -> Предикт
    """
    # 1. Готовим сырой датафрейм (исторический)
    full_df = prepare_raw_dataframe(weather_content, supplies_content, temperature_content)

    if isinstance(full_df, dict) and full_df.get('status') == 'error':
        return full_df

    try:
        # 2. Генерируем признаки на всей истории
        # (нам нужна история, чтобы посчитать скользящее среднее для "сегодня")
        df_feat = feature_engineering(full_df)

        # 3. Фильтруем: берем только ПОСЛЕДНЮЮ запись для каждого штабеля
        # Мы хотим предсказать риск на "сейчас"
        df_latest = df_feat.sort_values('date').groupby(['Склад', 'Штабель', 'Наим. ЕТСНГ']).tail(1).copy()

        # Отсекаем пустые/старые штабели (опционально, например где coal_mass=0)
        df_latest = df_latest[df_latest['coal_mass'] > 0.5]

        if df_latest.empty:
            return {'status': 'success', 'message': 'No active coal stacks found', 'top_risks': []}

        # 4. Label Encoding (Марка угля)
        if config.label_encoder:
            # Обработка неизвестных марок
            known_classes = set(config.label_encoder.classes_)
            df_latest['coal_type'] = df_latest['Наим. ЕТСНГ'].astype(str).apply(
                lambda x: config.label_encoder.transform([x])[0] if x in known_classes else 0
            )
        else:
            df_latest['coal_type'] = 0

        # 5. Формирование X для модели
        if config.ml_model:
            # Получаем фичи, которые ждет модель
            expected_features = config.ml_model.feature_names_in_

            # Проверка: есть ли все нужные колонки
            missing = [c for c in expected_features if c not in df_latest.columns]
            if missing:
                # Если каких-то нет, заполняем нулями
                for c in missing:
                    df_latest[c] = 0.0

            # Строгий порядок колонок
            X = df_latest[expected_features]

            # ПРЕДСКАЗАНИЕ
            probs = config.ml_model.predict_proba(X)[:, 1]  # Вероятность класса 1
            preds = config.ml_model.predict(X)

            df_latest['fire_probability'] = probs
            # Логика: либо по классу модели, либо жесткий порог вероятности
            df_latest['is_dangerous'] = (probs > 0.5).astype(int)
        else:
            df_latest['fire_probability'] = 0.0
            df_latest['is_dangerous'] = 0

        # 6. Формирование ответа
        # Оставляем читаемые поля
        result_cols = ['date', 'Склад', 'Штабель', 'Наим. ЕТСНГ', 'coal_mass',
                       'Максимальная температура', 'fire_probability', 'is_dangerous']
        result_df = df_latest[result_cols].copy()

        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')

        # Сортировка по риску
        top_risks = result_df.sort_values('fire_probability', ascending=False)

        return {
            'status': 'success',
            'total_stacks': len(result_df),
            'high_risk_count': int(result_df['is_dangerous'].sum()),
            'top_risks': top_risks.head(100).to_dict(orient='records'),
            'stats_by_warehouse': result_df.groupby('Склад')['is_dangerous'].sum().to_dict()
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': str(e)}


def export_data_prep(weather_content, supplies_content, temperature_content):
    """Вспомогательная функция для экспорта CSV (сырого, но объединенного)"""
    return prepare_raw_dataframe(weather_content, supplies_content, temperature_content)
