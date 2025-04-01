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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
    # Initialize sentiment analyzer from VADER
    analyzer = SentimentIntensityAnalyzer()

    # Define absolute words
    absolute_words = {"definitely", "too", "entire", "already", "totally", "only", "never", "always", "must"}

    # Define common interjections
    interjections = {"oh", "wow", "ugh", "huh", "aha", "oops", "ouch", "yeah", "right"}
    
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

        # Sentiment intensity
        sentiment_scores = analyzer.polarity_scores(text)
        sentiment_polarity = sentiment_scores['compound']
        sentiment_pos = sentiment_scores['pos']
        sentiment_neg = sentiment_scores['neg']
        sentiment_neu = sentiment_scores['neu']

        # Contrast between positive and negative sentiments
        sentiment_diff = abs(sentiment_pos - sentiment_neg)

        # Structural features
        num_adjectives = sum(1 for _, tag in tagged_tokens if tag.startswith("JJ"))
        num_adverbs = sum(1 for _, tag in tagged_tokens if tag.startswith("RB"))
        num_verbs = sum(1 for _, tag in tagged_tokens if tag.startswith("VB"))
        num_nouns = sum(1 for _, tag in tagged_tokens if tag.startswith("NN"))

        # Determining absolute words
        num_absolute_words = sum(word_counts[word] for word in absolute_words if word in word_counts)

        # Determining interjections
        num_interjections = sum(word_counts[word] for word in interjections if word in word_counts)
    
        features.append([
            num_adjectives, num_adverbs, num_verbs, num_nouns,
            num_exclamations, num_questions, num_quotes,
            num_absolute_words, num_interjections,
            sentiment_polarity, sentiment_pos, sentiment_neg, sentiment_neu,
            sentiment_diff
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
