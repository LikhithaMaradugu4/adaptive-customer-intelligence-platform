import os
import json
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


# ---------------------------------
# Load Dataset
# ---------------------------------

DATA_PATH = "data/training/intent_dataset.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


texts = [item["text"] for item in data]
labels = [item["intent"] for item in data]


# ---------------------------------
# Train/Test Split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


# ---------------------------------
# Build Pipeline
# ---------------------------------

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000
        )
    )
])


# ---------------------------------
# Train Model
# ---------------------------------

pipeline.fit(
    X_train,
    y_train
)


# ---------------------------------
# Evaluate Model
# ---------------------------------

predictions = pipeline.predict(X_test)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ---------------------------------
# Save Model
# ---------------------------------

os.makedirs(
    "app/models",
    exist_ok=True
)

joblib.dump(
    pipeline,
    "app/models/intent_classifier.pkl"
)

print("\nIntent model trained successfully.")
print("Saved to: app/models/intent_classifier.pkl")