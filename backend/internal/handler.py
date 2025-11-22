from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.params import File
from internal.service import process_data_pipeline

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

@router.get("/")
async def ping():
    return {"message":"pong"}