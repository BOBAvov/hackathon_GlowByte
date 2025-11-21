import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/ping")
async def ping():
    return {"message":"pong"}

if __name__ == "__main__":
    uvicorn.run(app,port=8080)