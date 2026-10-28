from pathlib import Path
import sqlite3
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="API for the Heart Attack Risk Prediction project",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_PATH = PROJECT_DIR / "heart_attack_prediction_india.csv"
MODEL_PATH = BASE_DIR / "best_model.pkl"
DB_PATH = BASE_DIR / "predictions.db"


df = pd.read_csv(DATA_PATH)


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Place best_model.pkl inside the backend folder."
    )

saved_data = joblib.load(MODEL_PATH)

model = saved_data["model"]
scaler = saved_data["scaler"]
columns = saved_data["columns"]



def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_data TEXT NOT NULL,
                prediction TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


init_db()


class PatientData(BaseModel):
    Age: float
    Gender: str
    State_Name: str
    Annual_Income: float
    Health_Insurance: str
    Diabetes: str
    Hypertension: str
    Obesity: str
    Smoking: str
    Alcohol_Consumption: str
    Physical_Activity: str
    Diet_Score: float
    Cholesterol_Level: float
    Triglyceride_Level: float
    LDL_Level: float
    HDL_Level: float
    Systolic_BP: float
    Diastolic_BP: float
    Air_Pollution_Exposure: str
    Family_History: str
    Stress_Level: str
    Healthcare_Access: str
    Emergency_Response_Time: float
    Heart_Attack_History: str



@app.get("/")
def home():
    return {
        "message": "Heart Attack Risk Prediction API is running"
    }



@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "database": str(DB_PATH)
    }


@app.get("/model-info")
def model_info():
    return {
        "model": type(model).__name__,
        "best_k": model.n_neighbors,
        "features": len(columns)
    }


@app.post("/predict")
def predict(patient: PatientData):
    try:
        # Convert submitted patient details into a DataFrame
        input_df = pd.DataFrame([patient.model_dump()])

        # Apply the same dummy encoding as training
        input_encoded = pd.get_dummies(
            input_df,
            drop_first=True
        )

        # Match training feature names and order
        input_encoded = input_encoded.reindex(
            columns=columns,
            fill_value=0
        )

        # Apply the previously fitted scaler
        input_scaled = scaler.transform(input_encoded)

        # Generate prediction
        result = model.predict(input_scaled)[0]
        prediction = str(result)

        # Save prediction history
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO predictions (input_data, prediction)
                VALUES (?, ?)
                """,
                (patient.model_dump_json(), prediction)
            )

        return {
            "prediction": prediction,
            "model": "K-Nearest Neighbors",
            "message": "Prediction generated successfully"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# View recent prediction history
@app.get("/history")
def prediction_history(limit: int = 10):
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 100"
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT id, prediction, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

    return {
        "history": [dict(row) for row in rows]
    }

