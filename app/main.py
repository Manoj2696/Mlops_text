from app.routes import router
from fastapi import FastAPI
from app.routes import router

app = FastAPI()

app.include_router(router)

@app.get("/")
def welcome():
    return {"message": "Welcome to the SVM Text Classification API"}
