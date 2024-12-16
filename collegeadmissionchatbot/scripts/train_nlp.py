import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

def preprocess_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)["dataset"]
    questions, intents = [], []
    for item in data:
        category = item["Category"]
        for question in item["User Questions"]:
            questions.append(question.lower())
            intents.append(category)
    return questions, intents

def train_intent_model(dataset_path, model_output_path):
    questions, intents = preprocess_dataset(dataset_path)
    model = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    model.fit(questions, intents)
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved at {model_output_path}")

if __name__ == "__main__":
    dataset_path = 'data/training_data/college_admissions.json'
    model_output_path = 'models/nlp/intent_model.pkl'
    train_intent_model(dataset_path, model_output_path)
