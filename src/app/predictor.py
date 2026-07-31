import time
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ==========================================
# Hugging Face Model Repository
# ==========================================

MODEL_NAME = "NivedJk/muril-hinglish-sentiment"

# ==========================================
# Device
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================
# Load Model
# ==========================================

print("Loading model from Hugging Face...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print("Model loaded successfully.")

# ==========================================
# Labels
# ==========================================

id_to_label = {
    0: "Negative 😡",
    1: "Neutral 😐",
    2: "Positive 😊",
}

# ==========================================
# Confidence Interpretation
# ==========================================

def get_confidence_level(confidence):

    if confidence >= 0.90:
        return "High Confidence"

    elif confidence >= 0.75:
        return "Good Confidence"

    elif confidence >= 0.60:
        return "Medium Confidence"

    else:
        return "Low Confidence"

# ==========================================
# Prediction
# ==========================================

def predict_sentiment(text: str):

    start_time = time.perf_counter()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

    inference_time = (
        time.perf_counter() - start_time
    )

    prediction = prediction.item()
    confidence_value = float(confidence.item())

    return {
        "label": id_to_label[prediction],
        "confidence": confidence_value,
        "confidence_level": get_confidence_level(
            confidence_value
        ),
        "inference_time_ms": inference_time * 1000,
        "probabilities": {
            "negative": float(probabilities[0][0]),
            "neutral": float(probabilities[0][1]),
            "positive": float(probabilities[0][2]),
        },
    }