# Support Ticket Auto-Categorizer

Reads a support ticket (subject + body) and predicts its category — `Billing`, `Technical`, `HR`, or `General` — so it can be auto-routed instead of a human triaging every incoming message.

## Approach

- **Data:** 60 hand-written support tickets (15 per category), stored in `tickets_dataset.csv`.
- **Preprocessing:** lowercase → strip URLs/emails/punctuation/numbers → remove stopwords.
- **Features:** TF-IDF, unigrams only (kept the vocabulary small and each feature well-observed, which gave steadier confidence scores than adding bigrams — see the notebook for why).
- **Model:** Multinomial Naive Bayes. Logistic Regression scored slightly higher on raw accuracy (83.3% vs 75%) but its probability outputs were nearly flat across all classes, which breaks the confidence threshold below. Naive Bayes was picked for usable, well-separated confidence — not accuracy alone.

## Bonus features implemented

- **Confidence score output** — every prediction returns a probability, not just a label.
- **60% "needs human review" threshold** — predictions below 60% confidence are *not* auto-assigned; they're routed to manual review with the model's best guess shown for context.
- **Priority tagging** — a keyword rule (`urgent`, `down`, `broken`, `asap`, `critical`, etc.) tags tickets `Urgent`/`Normal` independently of category.
- **Live demo** — `demo_cli.py` is a runnable CLI: type a ticket, get it classified instantly.
- **Reflection note** — Section 10 of the notebook.

## Files

- `ticket_categorizer.ipynb` — main notebook: load → clean → vectorize → train → evaluate → classify → live-demo pointer → reflection (run top to bottom).
- `tickets_dataset.csv` — the labeled training data.
- `pipeline.py` — the same logic as a plain script, useful for a quick local run without opening Jupyter.
- `demo_cli.py` — interactive CLI: `python3 demo_cli.py`, type a ticket, get category + confidence + priority back.

## Running it

```bash
pip install pandas scikit-learn matplotlib seaborn
jupyter notebook ticket_categorizer.ipynb
# or, for the live demo:
python3 demo_cli.py
```

## Approach summary (for the submission form)

Used TF-IDF (unigrams) + Multinomial Naive Bayes, chosen over Logistic Regression for well-separated confidence scores rather than raw accuracy alone. Predictions below 60% confidence aren't auto-assigned — they're routed to manual review with the best guess shown. Added a keyword-based Urgent/Normal priority tag and a runnable CLI demo (`demo_cli.py`) on top of the core classifier.
