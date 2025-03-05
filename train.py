import numpy as np
import pandas as pd
import joblib
import time

from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from prometheus_client import Counter, Histogram, generate_latest, start_http_server

# Load data and train model
csv_file = "legal_text_classification.csv"
df = pd.read_csv(csv_file)

df1 = df[['case_outcome', 'case_text']].copy()
df1 = df1[pd.notnull(df1['case_text'])]
df1.columns = ['Outcome', 'Text']

df2 = df1.sample(2500)
df2['category_id'] = df2['Outcome'].factorize()[0]

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df2.Text)
labels = df2.category_id

X_train, X_test, y_train, y_test = train_test_split(features, df2['category_id'], test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")

# Flask app
app = Flask(__name__)

# Load the model once
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Prometheus Metrics
PREDICTION_REQUESTS = Counter("prediction_requests_total", "Total number of prediction requests")
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Time taken for a prediction")

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    PREDICTION_REQUESTS.inc()

    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]

    PREDICTION_LATENCY.observe(time.time() - start_time)

    return jsonify({"text": text, "prediction": int(prediction)})

@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": "text/plain; charset=utf-8"}

if __name__ == "__main__":
    start_http_server(8000)  # Prometheus metrics on port 8000
    app.run(host="0.0.0.0", port=5000)
