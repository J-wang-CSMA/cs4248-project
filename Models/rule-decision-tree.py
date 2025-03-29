import pandas as pd
import numpy as np
import re
import string
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob

# Load dataset
CSV_FILE_PATH = os.path.join('Datasets', 'Sarcasm_Headlines_Dataset_v2.csv')
FEATURE = 'headline'
TARGET = 'is_sarcastic'

# Check if file exists
if not os.path.exists(CSV_FILE_PATH):
    raise FileNotFoundError(f"File not found at: {os.path.abspath(CSV_FILE_PATH)}")

# Read data
df = pd.read_csv(CSV_FILE_PATH)
headlines = df[FEATURE]
labels = df[TARGET]

# Split data
RANDOM_SEED = 42
train_texts, val_texts, train_labels, val_labels = train_test_split(
    headlines, labels, test_size=0.2, random_state=RANDOM_SEED
)

def extract_features(texts):
    features = []
    for text in texts:
        text = text.lower()
        num_exclamations = text.count("!")
        num_questions = text.count("?")
        num_quotes = text.count('"') + text.count("'")
        sentiment = TextBlob(text).sentiment.polarity
        
        # Presence of sarcastic keywords
        sarcasm_keywords = ["oh", "wow", "great", "yeah", "right"]
        keyword_count = sum(1 for word in sarcasm_keywords if word in text)
        
        features.append([num_exclamations, num_questions, num_quotes, sentiment, keyword_count])
    return np.array(features)

# Extract features
train_features = extract_features(train_texts)
val_features = extract_features(val_texts)

# Train Decision Tree classifier
clf = DecisionTreeClassifier(random_state=RANDOM_SEED)
clf.fit(train_features, train_labels)

# Predict and evaluate
val_preds = clf.predict(val_features)
accuracy = accuracy_score(val_labels, val_preds)
report = classification_report(val_labels, val_preds)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)
