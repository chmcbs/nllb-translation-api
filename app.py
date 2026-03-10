"""Multi-language FastAPI translation API using Facebook NLLB"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Gauge
import psutil
import json

with open("config.json") as f:
    CONFIG = json.load(f)

SUPPORTED_LANGUAGES = CONFIG["supported_languages"]
TRANSLATION_MODEL = CONFIG["translation_model"]
MIN_STR_LENGTH = CONFIG["min_str_length"]
MAX_STR_LENGTH = CONFIG["max_str_length"]
MAX_NEW_TOKENS = CONFIG["max_new_tokens"]

CPU_USAGE = Gauge("system_cpu_usage_percent", "Current CPU usage percentage")
MEMORY_USAGE = Gauge("system_memory_usage_percent", "Current memory usage percentage")

class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=MIN_STR_LENGTH, max_length=MAX_STR_LENGTH)
    source: str
    target: str
    
    @field_validator("source", "target")
    @classmethod
    def validate_language(cls, v: str):
        if v.lower() not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {v}. Supported: {list(SUPPORTED_LANGUAGES.keys())}")
        return v.lower()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload the translator before the app starts"""
    app.state.loaded = False
    print("Loading translator...")
    try:
        app.state.translator = pipeline("translation", model=TRANSLATION_MODEL)
        print("Translator loaded successfully.")
        app.state.loaded = True
    except Exception as e:
        print(f"Error loading translator: {e}")
        raise
    yield

app = FastAPI(lifespan=lifespan)

def update_system_metrics(info):
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)

Instrumentator().instrument(app).add(metrics.default()).add(update_system_metrics).expose(app)

def translate_text(text: str, source: str, target: str):
    translation = app.state.translator(text, src_lang=source, tgt_lang=target, max_new_tokens=MAX_NEW_TOKENS)
    return translation[0]["translation_text"]

@app.post("/translate")
def translate(request: TranslateRequest):
    if not hasattr(app.state, "translator"):
        raise HTTPException(status_code=503, detail="Translator not loaded")
    return {"translation": translate_text(request.text, SUPPORTED_LANGUAGES[request.source], SUPPORTED_LANGUAGES[request.target])}

@app.get("/languages")
def languages():
    return list(SUPPORTED_LANGUAGES.keys())

@app.get("/ready")
def ready():
    if getattr(app.state, "loaded", False) and hasattr(app.state, "translator"):
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Translator not loaded")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Internal server error"})