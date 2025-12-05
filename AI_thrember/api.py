from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from predictor import AIMaPPredictor
from fastapi.middleware.cors import CORSMiddleware

# ===============================
# RATE LIMITING
# ===============================
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import PlainTextResponse

limiter = Limiter(key_func=get_remote_address)

# ===============================
# APP INIT
# ===============================
app = FastAPI()
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request, exc):
    return PlainTextResponse("Too many requests. Slow down.", status_code=429)

# ===============================
# CORS
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # change to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Predictor + Security Constants
# ===============================
predictor = AIMaPPredictor()

MAX_FILE_SIZE = 300 * 1024 * 1024   # 300 MB

# Allowed file extensions
ALLOWED_EXTENSIONS = {".exe", ".dll", ".sys", ".pdf"}


# ===============================
# PREDICT ENDPOINT (Rate Limited)
# ===============================
@app.post("/predict")
@limiter.limit("5/minute")    # 5 scans per minute per IP
async def predict_file(request: Request, file: UploadFile = File(...)):

    filename = file.filename.lower()

    # ------------------------------
    # EXTENSION VALIDATION
    # ------------------------------
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed extensions: .exe, .dll, .sys, .pdf, .elf"
        )

    # ------------------------------
    # READ FILE
    # ------------------------------
    bytez = await file.read()

    # SIZE LIMIT CHECK
    if len(bytez) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed size is {MAX_FILE_SIZE // (1024*1024)} MB."
        )

    # ------------------------------
    # RUN PREDICTION
    # ------------------------------
    result = predictor.predict(bytez)
    result["filename"] = filename

    return result
