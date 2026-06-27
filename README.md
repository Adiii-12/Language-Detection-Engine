# LanguagePredictor: Locally-Run NLP Engine for High-Accuracy Language Detection 🌍

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5" />
</p>

An edge-computing natural language processing application that utilizes an optimized **LinearSVC** pipeline to detect and classify text across multiple languages in real-time. Engineered strictly for localized execution to ensure zero-latency inference and total data privacy.

---

## ⚙️ System Architecture

Unlike heavy cloud-dependent APIs, `LanguagePredictor` processes incoming queries on-device. The pipeline is structured to execute seamlessly within lightweight computing environments:

---

## 🚀 Why This Architecture? (Model Performance)

* **98.91% Validation Accuracy:** Through rigorous iterative testing, the classification model transitioned from a Naive Bayes baseline to an advanced Linear Support Vector Classifier (LinearSVC) to hit peak predictive power.
* **Short-Phrase Resilience:** The feature extraction pipeline specifically addresses ambient noise and short text snippets by utilizing character-level n-grams instead of plain words.
* **Zero Latency:** Running entirely locally via Flask means user data never leaves the machine. No network handshake overhead or third-party API dependencies.
* **Offline Deployment:** Built using highly optimized vectorizers from `scikit-learn` to maintain a low local memory footprint, making it ideal for offline integrations.

---

## 🛠️ Project Structure

```text
├── templates/
│   └── index.html          # Lightweight, clean Web UI for user interaction
├── .gitattributes          # Language bar optimization configuration
├── language.csv            # Standardized training dataset
├── main.py                 # Core Flask backend & Machine Learning inference engine
└── restore_dataset.py      # Dataset pipeline management script

---

## 👥 Author

* **Created & Developed by: Aditya Pratap Singh Sisodiya
