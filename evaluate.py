import json
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load data
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# find text column
text_col = "text"
if text_col not in fake.columns or text_col not in real.columns:
    for c in ("text", "title", "headline", "content"):
        if c in fake.columns and c in real.columns:
            text_col = c
            break
    else:
        raise ValueError("Could not find a shared text column in CSVs")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real]).reset_index(drop=True)
texts = data[text_col].astype(str).tolist()
labels = data["label"].astype(int).tolist()

# same split as training
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# load model and tokenizer from ./model
if not os.path.isdir("model"):
    raise FileNotFoundError("model/ directory not found. Make sure the trained model is saved in ./model")

tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForSequenceClassification.from_pretrained("model")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# batch inference
batch_size = 32
all_preds = []
all_probs = []
with torch.no_grad():
    for i in range(0, len(val_texts), batch_size):
        batch_texts = val_texts[i : i + batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt", max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=1).tolist()
        all_preds.extend(preds)
        all_probs.extend(probs.tolist())

# metrics
acc = accuracy_score(val_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(val_labels, all_preds, average="binary")
cm = confusion_matrix(val_labels, all_preds)

# confidence analysis
all_probs = np.array(all_probs)
pred_probs = np.max(all_probs, axis=1)  # max prob for predicted class
avg_confidence = np.mean(pred_probs)
min_confidence = np.min(pred_probs)
max_confidence = np.max(pred_probs)

print(f"Average confidence: {avg_confidence:.4f}")
print(f"Min confidence: {min_confidence:.4f}")
print(f"Max confidence: {max_confidence:.4f}")
print(f"Confidence distribution: {np.histogram(pred_probs, bins=5)[0]}")

metrics = {
    "accuracy": float(acc),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "n_val": len(val_labels),
    "avg_confidence": float(avg_confidence),
    "min_confidence": float(min_confidence),
    "max_confidence": float(max_confidence),
}

# save metrics
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# save confusion matrix
cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]) 
cm_df.to_csv("confusion.csv")

print("Evaluation completed")
print(json.dumps(metrics, indent=2))
print("Confusion matrix saved to confusion.csv")
