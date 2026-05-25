from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, List
import joblib
import pandas as pd
import os
from datetime import datetime
import logging
from contextlib import asynccontextmanager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Глобальные переменные
model = None
preprocessor = None

class CreditApplication(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    loan_duration: int = Field(..., ge=1, le=360)
    employment_years: int = Field(..., ge=0, le=50)
    credit_history: str = Field(...)
    purpose: str = Field(...)
    savings: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    default_probability: float
    risk_category: str
    prediction: int
    timestamp: str
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    version: str = "1.0.0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor
    
    logger.info("Загрузка модели...")
    try:
        model_path = os.getenv('MODEL_PATH', 'artifacts/credit_risk_model.pkl')
        preprocessor_path = os.getenv('PREPROCESSOR_PATH', 'artifacts/preprocessor.pkl')
        
        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            logger.info("✅ Модель и препроцессор успешно загружены")
        else:
            logger.warning(f"⚠️ Модель не найдена по пути: {model_path}")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
    
    yield
    logger.info("Выключение сервера...")

app = FastAPI(
    title="Credit Risk Scoring API",
    description="Сервис для оценки кредитного риска",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {"message": "Credit Risk Scoring API", "docs": "/docs", "health": "/health"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(application: CreditApplication):
    logger.info(f"Получен запрос: {application.dict()}")
    
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Модель не загружена. Обучите модель через python src/models/train.py"
        )
    
    try:
        input_data = pd.DataFrame([application.dict()])
        X_processed = preprocessor.transform(input_data)
        probability = float(model.predict_proba(X_processed)[0, 1])
        prediction = int(probability >= 0.5)
        
        if probability < 0.3:
            risk_category = "low"
        elif probability < 0.7:
            risk_category = "medium"
        else:
            risk_category = "high"
        
        return PredictionResponse(
            default_probability=probability,
            risk_category=risk_category,
            prediction=prediction,
            timestamp=datetime.now().isoformat(),
            status="success"
        )
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)