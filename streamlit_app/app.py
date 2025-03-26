from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import os

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load model and preprocessor
model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            Age: float = Form(...),
            BMI: float = Form(...),
            Exercise_Frequency: int = Form(...),
            Diet_Quality: float = Form(...),
            Sleep_Hours: float = Form(...),
            Smoking_Status: int = Form(...),
            Alcohol_Consumption: float = Form(...)):

    # Convert form data to DataFrame
    input_data = pd.DataFrame([{
        "Age": Age,
        "BMI": BMI,
        "Exercise_Frequency": Exercise_Frequency,
        "Diet_Quality": Diet_Quality,
        "Sleep_Hours": Sleep_Hours,
        "Smoking_Status": Smoking_Status,
        "Alcohol_Consumption": Alcohol_Consumption
    }])

    # Preprocess and predict
    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)[0]

    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": round(prediction, 2)
    })
