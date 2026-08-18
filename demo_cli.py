"""
demo_cli.py
-----------
A tiny live demo: type in a ticket, get it categorized instantly.
Turns the notebook's pipeline into something you can actually run and poke at,
not just read.

Usage:
    python3 demo_cli.py
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB

CONFIDENCE_THRESHOLD = 0.60
URGENT_KEYWORDS = {
    "urgent", "asap", "immediately", "down", "broken", "crash", "crashed",
    "critical", "emergency", "outage", "not working", "stuck", "failing", "failed",
}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in ENGLISH_STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


def get_priority(raw_text: str) -> str:
    t = raw_text.lower()
    return "Urgent" if any(kw in t for kw in URGENT_KEYWORDS) else "Normal"


def train():
    df = pd.read_csv("tickets_dataset.csv")
    df["clean_text"] = (df["subject"] + " " + df["body"]).apply(clean_text)

    vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, max_df=0.85, sublinear_tf=True)
    X = vectorizer.fit_transform(df["clean_text"])

    model = MultinomialNB(alpha=0.05)
    model.fit(X, df["category"])
    return vectorizer, model


def classify(vectorizer, model, subject: str, body: str) -> dict:
    raw_text = subject + " " + body
    clean = clean_text(raw_text)
    probs = model.predict_proba(vectorizer.transform([clean]))[0]
    classes = model.classes_
    best_idx = np.argmax(probs)
    best_label, best_conf = classes[best_idx], float(probs[best_idx])
    auto_assigned = best_conf >= CONFIDENCE_THRESHOLD

    return {
        "predicted_category": best_label,
        "confidence": round(best_conf, 3),
        "priority": get_priority(raw_text),
        "auto_assigned": auto_assigned,
    }


def main():
    print("Training classifier on tickets_dataset.csv ...")
    vectorizer, model = train()
    print("Ready. Type a ticket to classify it, or 'quit' to exit.\n")

    while True:
        subject = input("Subject: ").strip()
        if subject.lower() in ("quit", "exit"):
            break
        body = input("Body: ").strip()

        result = classify(vectorizer, model, subject, body)
        print(f"\n  Category:   {result['predicted_category']}")
        print(f"  Confidence: {result['confidence']:.0%}")
        print(f"  Priority:   {result['priority']}")
        if result["auto_assigned"]:
            print(f"  Status:     Auto-assigned to {result['predicted_category']}\n")
        else:
            print(f"  Status:     Below {CONFIDENCE_THRESHOLD:.0%} confidence - routed to manual review\n")


if __name__ == "__main__":
    main()
