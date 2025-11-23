from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.params import File, Query
from internal.service import process_data_pipeline, export_data_prep, save_file, get_saved_data
import io

router = APIRouter()


@router.get('/ping')
async def ping():
    return {"status": "ok", "message": "pong"}

@router.post('/import')
async def import_data(
        weather_file: UploadFile = File(...),
        supplies_file: UploadFile = File(...),
        temperature_file: UploadFile = File(...)
):
    try:
        save_file('weather.csv', await weather_file.read())
        save_file('supplies.csv', await supplies_file.read())
        save_file('temperature.csv', await temperature_file.read())
        return {"status": "success", "message": "Files uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/predict')
async def predict_fire_risk():
    try:
        w_content = get_saved_data('weather.csv')
        s_content = get_saved_data('supplies.csv')
        t_content = get_saved_data('temperature.csv')

        result = process_data_pipeline(w_content, s_content, t_content)

        if result.get('status') == 'error':
            raise HTTPException(status_code=500, detail=result['message'])

        return JSONResponse(status_code=200, content=result)

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail='Data files missing. Use /api/import first.')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/export')
async def export_data(format: str = Query('csv')):
    try:
        w_content = get_saved_data('weather.csv')
        s_content = get_saved_data('supplies.csv')
        t_content = get_saved_data('temperature.csv')

        # Предполагаем, что эта функция у вас есть из прошлого кода
        df = export_data_prep(w_content, s_content, t_content)

        if isinstance(df, dict) and df.get('status') == 'error':
            raise HTTPException(status_code=500, detail='Failed to prepare data')

        buffer = io.BytesIO()

        if format == "xlsx":
            # Требуется: pip install openpyxl
            df.to_excel(buffer, index=False, engine='openpyxl')
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = "export.xlsx"
        else:
            df.to_csv(buffer, index=False, encoding='utf-8')
            media_type = "text/csv"
            filename = "export.csv"

        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type=media_type,
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
