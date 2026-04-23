from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

MODEL_DIR = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_news(text):
    text_lower = text.lower()

    # ✅ STRONG REAL NEWS RULES
    real_keywords = [
        "nasa", "isro", "government", "policy", "scientists",
        "research", "study", "announces", "launch", "report"
    ]

    # ✅ STRONG FAKE NEWS RULES
    fake_keywords = [
        "aliens", "secret", "shocking", "cure", "forward this",
        "hidden truth", "100% cure", "miracle", "conspiracy"
    ]

    # Rule-based detection
    if any(word in text_lower for word in real_keywords):
        return "Real News", 0.90

    if any(word in text_lower for word in fake_keywords):
        return "Fake News", 0.95

    # 🤖 Model prediction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

    fake_prob = probs[0]
    real_prob = probs[1]

    confidence = max(fake_prob, real_prob)

    # ✅ FINAL SAFETY LOGIC
    if confidence < 0.7:
        return "Uncertain", confidence

    # If model is too biased → reduce fake dominance
    if fake_prob > 0.9:
        return "Uncertain", fake_prob

    label = "Real News" if real_prob > fake_prob else "Fake News"

    return label, confidence