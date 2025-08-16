# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
# load trained model
model = joblib.load("sms_spam_model.pkl")

# define request schema
class Message(BaseModel):
    text: str
    threshold: float = 0.50   # optional, default 0.5

LABELS = {0: "ham", 1: "spam"}

# create app
app = FastAPI(title="SMS Spam Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for local demo; restrict in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "SMS Spam Detector API is running"}

@app.post("/predict")
def predict(msg: Message):
    prob_spam = float(model.predict_proba([msg.text])[0][1])
    pred = int(prob_spam >= msg.threshold)
    return {
        "input": msg.text,
        "label": LABELS[pred],
        "spam_probability": prob_spam,
        "threshold": msg.threshold
    }
