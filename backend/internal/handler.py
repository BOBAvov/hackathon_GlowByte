from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.params import File, Query
from internal.service import process_data_pipeline, prepare_data
import io

router = APIRouter()

@router.post("/predict-fire-risk")
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


@router.post("/export")
async def export_data(
    format: str = Query("csv"),
    weather_file: UploadFile = File(...),
    supplies_file: UploadFile = File(...),
    temperature_file: UploadFile = File(...)
):
    w_content = await weather_file.read()
    s_content = await supplies_file.read()
    t_content = await temperature_file.read()

    df = prepare_data(w_content, s_content, t_content)

    if isinstance(df, dict) and df.get("status") == "error":
        raise HTTPException(
            status_code=500,
            detail="Failed to prepare data"
        )
    
    buffer = io.BytesIO()
    if format == "csv":
        df.to_csv(buffer, index=False, encoding="utf-8")
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=export.csv"
            }
        )
    
    if format == "xlsx":
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": "attachment; filename=export.xlsx"
            }
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file format: {format}"
    )


@router.get("/")
async def ping():
    return {"message":"pong"}