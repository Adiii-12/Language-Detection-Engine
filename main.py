import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

if os.path.exists("language.csv"):
    csv_path = "language.csv"
else:
    csv_path = "../language.csv"

# --- BULLETPROOF SWEET-SPOT PIPELINE ---
data = pd.read_csv(csv_path)

# Standardize column headers to lowercase
data.columns = data.columns.str.lower()

data.dropna(subset=['text', 'language'], inplace=True)
data.drop_duplicates(subset=['text'], inplace=True)

x = np.array(data['text'])
y = np.array(data['language'])

# Word-boundary char n-grams — stops English bleeding into Asian language classes
cv = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True, min_df=2)
X = cv.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model = LogisticRegression(C=1.8, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
calculated_accuracy = accuracy_score(y_test, y_pred) * 100

# Common English words for short-phrase fallback
COMMON_ENGLISH = {
    "how", "are", "you", "hello", "hi", "good", "morning", "day",
    "what", "is", "this", "ok", "okay", "yes", "no", "bye", "thanks",
    "thank", "please", "sorry", "hey", "i", "am", "my", "name", "the",
    "a", "an", "it", "to", "do", "be", "in", "on", "of", "and", "or"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analytics', methods=['GET'])
def get_analytics():
    lang_counts = data['language'].value_counts()
    return jsonify({
        'accuracy': f"{calculated_accuracy:.2f}%",
        'test_samples': len(y_test),
        'train_samples': len(X_train.toarray()) if hasattr(X_train, 'toarray') else len(y_train),
        'total_samples': len(data),
        'total_languages': int(data['language'].nunique()),
        'languages': lang_counts.index.tolist()[:20]
    })

@app.route('/predict', methods=['POST'])
def predict():
    req_data = request.get_json()
    user_text = req_data.get('text', '')

    if not user_text.strip():
        return jsonify({'error': 'Please enter some text.'}), 400

    # Short English phrase fallback
    words = [w.strip("?,.!;:'\"").lower() for w in user_text.split()]
    if words and all(w in COMMON_ENGLISH for w in words):
        return jsonify({'language': 'English', 'confidence': 99.9})

    user_vector = cv.transform([user_text])
    output = model.predict(user_vector)

    # Get confidence probabilities
    proba = model.predict_proba(user_vector)[0]
    confidence = float(max(proba)) * 100

    final_output = str(output[0]).capitalize()
    return jsonify({'language': final_output, 'confidence': round(confidence, 1)})

if __name__ == '__main__':
    app.run(debug=True, port=9000)