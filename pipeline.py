"""
pipeline.py
-----------
End-to-end draft: load -> clean -> vectorize -> train -> evaluate -> classify.
Used to sanity-check everything before wiring it into the notebook.
"""

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("tickets_dataset.csv")
df["text"] = df["subject"] + " " + df["body"]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          
    text = re.sub(r"\S+@\S+", " ", text)                   
    text = re.sub(r"[^a-z\s]", " ", text)                  
    text = re.sub(r"\s+", " ", text).strip()                
    tokens = [w for w in text.split() if w not in ENGLISH_STOP_WORDS and len(w) > 1]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["category"],
    test_size=0.2, random_state=42, stratify=df["category"]
)


vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, max_df=0.85, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


nb = MultinomialNB(alpha=0.05)
nb.fit(X_train_vec, y_train)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_vec, y_train)

nb_acc = accuracy_score(y_test, nb.predict(X_test_vec))
lr_acc = accuracy_score(y_test, lr.predict(X_test_vec))
print(f"Naive Bayes accuracy: {nb_acc:.3f}")
print(f"Logistic Regression accuracy: {lr_acc:.3f}")


model = nb
print(f"Selected model: {model.__class__.__name__} (chosen for usable confidence separation, not just raw accuracy)")


y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=sorted(df["category"].unique())))


CONFIDENCE_THRESHOLD = 0.60  

URGENT_KEYWORDS = {
    "urgent", "asap", "immediately", "down", "broken", "crash", "crashed",
    "critical", "emergency", "outage", "not working", "stuck", "failing", "failed",
}

def get_priority(raw_text: str) -> str:
    """Simple keyword-rule priority tag, independent of the category model."""
    t = raw_text.lower()
    return "Urgent" if any(kw in t for kw in URGENT_KEYWORDS) else "Normal"

def classify_ticket(subject: str, body: str) -> dict:
    """Classify a single new ticket in real time.

    Always returns the model's best-guess category and its confidence score.
    If confidence is below CONFIDENCE_THRESHOLD, the ticket is NOT auto-assigned -
    it's marked for manual review instead (best guess is still shown for the
    reviewer's convenience). Also tags Urgent/Normal priority via keyword rules.
    """
    raw_text = subject + " " + body
    clean = clean_text(raw_text)
    vec = vectorizer.transform([clean])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    best_idx = np.argmax(probs)
    best_label, best_conf = classes[best_idx], float(probs[best_idx])
    auto_assigned = best_conf >= CONFIDENCE_THRESHOLD

    return {
        "predicted_category": best_label,
        "confidence": round(best_conf, 3),
        "priority": get_priority(raw_text),
        "auto_assigned": auto_assigned,
        "status": (
            f"Auto-assigned to {best_label}"
            if auto_assigned
            else f"Needs manual review (best guess: {best_label})"
        ),
    }

sample_tickets = [
    ("Can't log into my account", "I keep getting an invalid credentials error every time I try to log in. This is urgent, I have a submission deadline in an hour and can't access anything."),
    ("Billed twice this month", "I was charged for my subscription twice this month on the same card, once on the 1st and again on the 4th. Please refund the duplicate charge."),
    ("Question on sick leave balance", "Could someone confirm how many sick leave days I have remaining for this year? I want to plan a few days off next month."),
    ("Interested in partnering with you", "Our agency works with a lot of small businesses and we think a referral partnership with your platform could work well for both sides."),
    ("Not sure who to ask about this", "Something felt a bit off with my account today but I'm not really sure what department this should go to."),
]

print("\n--- Sample predictions ---")
for subj, body in sample_tickets:
    r = classify_ticket(subj, body)
    print(f"[{r['predicted_category']:9s} | conf={r['confidence']:.2f} | {r['priority']:6s}] {subj}")
    print(f"   -> {r['status']}")
