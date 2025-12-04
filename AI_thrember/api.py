from fastapi import FastAPI, UploadFile
from predictor import AIMaPPredictor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins (simple setup)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # allow all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = AIMaPPredictor()

@app.post("/predict")
async def predict_file(file: UploadFile):
    bytez = await file.read()
    result = predictor.predict(bytez)
    return result
