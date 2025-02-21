import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, jsonify
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



csv_file = "legal_text_classification.csv"  # Ensure this is in your container
df = pd.read_csv(csv_file)



df1 = df[['case_outcome', 'case_text']].copy()

# Remove missing values (NaN)
df1 = df1[pd.notnull(df1['case_text'])]

# Renaming second column for a simpler name
df1.columns = ['Outcome', 'Text'] 

df2 = df1.sample(5000)
# Preprocessing of the data using tfidf
df2['category_id'] = df2['Outcome'].factorize()[0]
category_id_df = df2[['Outcome', 'category_id']].drop_duplicates()

category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Outcome']].values)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2), 
                        stop_words='english')

features = tfidf.fit_transform(df2.Text).toarray()
labels = df2.category_id

# train test split
X, y = features, df2['category_id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fitting the model
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")



# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")

# Flask API for predictions
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]

    return jsonify({"text": text, "prediction": int(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)