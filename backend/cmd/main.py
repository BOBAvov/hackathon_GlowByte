from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from internal import config
from internal.handler import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    print("🚀 Запуск приложения...")

    config.loadConfig()
    ml_model = config.ml_model

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

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)