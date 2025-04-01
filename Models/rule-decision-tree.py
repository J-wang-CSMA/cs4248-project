import pandas as pd
import numpy as np
import os
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

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
        tokens = word_tokenize(text)
        tagged_tokens = pos_tag(tokens)
        word_counts = Counter(tokens)

        # Punctuations
        num_exclamations = text.count("!")
        num_questions = text.count("?")
        num_quotes = "'" in text

        # Structural features
        num_adjectives = sum(1 for _, tag in tagged_tokens if tag.startswith("JJ"))
        num_adverbs = sum(1 for _, tag in tagged_tokens if tag.startswith("RB"))
        num_verbs = sum(1 for _, tag in tagged_tokens if tag.startswith("VB"))
        num_nouns = sum(1 for _, tag in tagged_tokens if tag.startswith("NN"))
    
        features.append([
            num_adjectives, num_adverbs, num_verbs, num_nouns,
            num_exclamations, num_questions, num_quotes,
        ])
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
