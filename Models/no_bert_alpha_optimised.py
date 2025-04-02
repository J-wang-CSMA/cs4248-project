"""
Sarcasm Headline Detection Pipeline

This script implements a complete pipeline for detecting sarcasm in news headlines.
It includes:
1.  Setup of NLP resources (spaCy, VADER, NRCLex).
2.  Definition of keyword lists and patterns for feature engineering.
3.  A comprehensive set of feature extraction functions covering lexical, POS,
    sentiment, emotion, exaggeration, mundanity, NER, structural incongruity,
    and interaction features.
4.  Data loading, cleaning, feature extraction, imputation, and optional saving/loading
    of pre-computed features.
5.  Optional Principal Component Analysis (PCA) for dimensionality reduction.
6.  Training of individual machine learning models (Logistic Regression, Decision Tree,
    Random Forest, PyTorch MLP, XGBoost, SVM).
7.  Training of ensemble models (Voting Classifiers based on individual models,
    a dedicated 3x MLP ensemble, and a Stacking ensemble).
8.  Evaluation of all trained models using standard classification metrics.
9.  Display of feature importances for interpretable models.
"""

# --- Core Libraries ---
import os
import re
import time
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import nltk
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Check if NRCLex is installed
try:
    from nrclex import NRCLex
except ImportError:
    print("ERROR: NRCLex not found. Please install it: pip install NRCLex")
    exit()

from scipy.spatial.distance import cosine
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Keeping StandardScaler for now
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from torch.utils.data import DataLoader, TensorDataset

# Check if tqdm is installed for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Check if XGBoost is installed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# --- Configuration ---
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning, module='nrclex') # Optional

# --- Constants ---
FEATURES_X_CSV = 'optimized_features_X.csv'
TARGET_Y_CSV = 'target_y.csv'

# --- Keyword Lists & Regex Patterns (Global Constants) ---
# (Keyword lists and compiled regex remain unchanged - keeping for brevity)
INTENSIFIER_KEYWORDS = {
    "absolutely", "completely", "entirely", "extremely", "fucking", "fully", "goddamn", "highly", "hugely",
    "incredibly", "insanely", "literally", "massive", "perfectly", "positively", "purely", "remarkably",
    "strikingly", "stupendously", "terrifically", "thoroughly", "totally", "unbelievably", "utterly", "very",
    "awfully", "certainly", "clearly", "considerably", "damn", "deadly", "decidedly", "deeply", "distinctly",
    "eminently", "enormously", "especially", "exceptionally", "extraordinarily", "fairly", "frightfully",
    "greatly", "hella", "immensely", "indeed", "intensely", "jolly", "mightily", "most", "noticeably",
    "particularly", "peculiarly", "plenty", "pretty", "quite", "radically", "rather", "real", "really",
    "significantly", "so", "somewhat", "strongly", "substantially", "super", "supremely", "surely", "terribly",
    "truly", "unusually", "vastly", "almost", "barely", "effectively", "essentially", "fundamentally", "hardly",
    "just", "largely", "mainly", "merely", "minimally", "mostly", "nearly", "nominally", "only", "partially",
    "partly", "practically", "primarily", "principally", "relatively", "roughly", "scarcely", "simply", "slightly",
    "technically", "virtually", "always", "constantly", "definitely", "every", "never", "undoubtedly",
    "unquestionably", "bloody", "crazy", "damn", "freaking", "mad", "wicked"
}
GENERIC_PERSON_SINGLE_WORDS = {
    "man", "woman", "person", "guy", "gal", "dude", "chap", "bloke", "individual", "adult", "someone", "somebody",
    "anyone", "anybody", "everyone", "everybody", "nobody", "mom", "dad", "mother", "father", "parent", "parents",
    "son", "daughter", "child", "children", "kid", "kids", "baby", "infant", "toddler", "grandma", "grandpa",
    "grandmother", "grandfather", "grandparent", "husband", "wife", "spouse", "brother", "sister", "sibling",
    "uncle", "aunt", "cousin", "friend", "buddy", "pal", "acquaintance", "stranger", "teen", "teenager", "youth",
    "senior", "retiree", "coworker", "colleague", "boss", "employee", "staffer", "worker", "customer", "client",
    "patient", "student", "teacher", "driver", "pedestrian", "bystander", "resident", "neighbor"
}
GENERIC_PERSON_PHRASES_REGEX_PATTERNS = [
    r'\barea\s+(man|woman|dad|mom|resident|teenager?|youth|official|couple|child|business\s+owner|teacher)\b',
    r'\blocal\s+(man|woman|resident|teenager?|youth|official|couple|child|business\s+owner|teacher|mom|dad)\b',
    r'\b(city|town|county|nearby|neighborhood)\s+(resident|man|woman)\b', r'\bsenior\s+citizen\b',
    r'\belderly\s+(man|woman|couple|person)\b', r'\bmiddle-aged\s+(man|woman|person)\b',
    r'\byoung\s+(child|man|woman|person)\b',
    r'\b(office|store|shop|fast\s+food|restaurant|government|city|state|county|hospital|health\s+care)\s+(worker|employee)\b',
    r'\b(delivery|bus|taxi|truck)\s+(driver|person)\b',
    r'\b(unnamed|anonymous)\s+(official|source|person|employee|worker)\b', r'\b(pet|car|home)\s*owner\b',
    r'\beyewitness(?:es)?\b', r'\bconcerned\s+(citizen|resident|parent|person)\b',
    r'\bmember\s+of\s+the\s+public\b', r'\b(injured|missing)\s+(man|woman|person|teen|child)\b',
    r'\bmarried\s+couple\b', r'\bgroup\s+of\s+(friends|teens|youths|people|students|workers)\b',
    r'\bfamily\s+members?\b', r'\b(several|some|few|many)\s+people\b', r'\bno\s+one\b',
]
GENERIC_PERSON_REGEX_COMPILED = [re.compile(pattern, re.IGNORECASE) for pattern in GENERIC_PERSON_PHRASES_REGEX_PATTERNS]
MUNDANE_ACTION_VERBS = {
    "use", "enter", "announce", "learn", "stare", "hand", "visit", "watch", "find", "get", "take", "make", "keep",
    "report", "say", "tell", "ask", "talk", "speak", "call", "greet", "wave", "nod", "write", "read", "email",
    "text", "post", "tweet", "see", "hear", "feel", "smell", "taste", "think", "believe", "know", "realize",
    "consider", "remember", "forget", "notice", "wonder", "decide", "plan", "hope", "wish", "expect", "assume",
    "guess", "go", "come", "leave", "arrive", "stay", "sit", "stand", "lie", "walk", "run", "move", "turn", "carry",
    "hold", "put", "place", "set", "give", "bring", "send", "eat", "drink", "sleep", "wake", "breathe", "look",
    "point", "reach", "touch", "open", "close", "start", "stop", "continue", "wait", "try", "help", "have", "own",
    "need", "want", "like", "love", "hate", "prefer", "seem", "appear", "become", "remain", "work", "play", "study",
    "shop", "buy", "sell", "pay", "drive", "ride", "cook", "clean", "wash", "dress", "undress", "sit", "add", "begin",
    "change", "check", "choose", "finish", "happen", "include", "let", "listen", "live", "lose", "mean", "meet",
    "offer", "order", "pass", "pull", "push", "raise", "receive", "return", "serve", "show", "spend", "suggest", "wear"
}

# --- NLP Resources (Initialized in setup_nlp_resources) ---
nlp: Optional[spacy.language.Language] = None
vader_analyzer: Optional[SentimentIntensityAnalyzer] = None


# --- 1. Setup ---

def setup_nlp_resources() -> None:
    """
    Loads the spaCy model and initializes the VADER sentiment analyzer.

    Downloads necessary resources (spaCy model, VADER lexicon) if they are not
    found locally. Populates the global `nlp` and `vader_analyzer` variables.
    """
    global nlp, vader_analyzer
    print("--- Setting up NLP Resources ---")

    # Load spaCy Model
    spacy_model_name = "en_core_web_lg"
    try:
        nlp = spacy.load(spacy_model_name)
        print(f"Loaded spaCy model '{spacy_model_name}'")
    except OSError:
        print(f"Downloading spaCy model '{spacy_model_name}'...")
        try:
            spacy.cli.download(spacy_model_name)
            nlp = spacy.load(spacy_model_name)
            print("Model downloaded and loaded.")
        except Exception as e:
            print(f"ERROR: Failed to download or load spaCy model '{spacy_model_name}': {e}")
            exit() # Critical resource, exit if unavailable

    # Initialize VADER
    try:
        # Try initializing first (might work if lexicon downloaded previously)
        vader_analyzer = SentimentIntensityAnalyzer()
        # Test if lexicon is available by running it once
        _ = vader_analyzer.polarity_scores("test")
        print("Initialized VADER Analyzer.")
    except LookupError:
        print("VADER lexicon not found. Downloading...")
        try:
            nltk.download('vader_lexicon')
            vader_analyzer = SentimentIntensityAnalyzer()
            print("VADER lexicon downloaded and analyzer initialized.")
        except Exception as e:
            print(f"ERROR: Failed to download VADER lexicon: {e}")
            exit() # Critical resource, exit if unavailable
    except Exception as e:
        print(f"ERROR: Failed to initialize VADER Analyzer: {e}")
        exit()

    print("NLP resources setup complete.")


# --- 2. Feature Extraction Functions ---

def get_lexical_basic_features(doc: spacy.tokens.doc.Doc) -> Dict[str, Union[int, float]]:
    """
    Extracts basic lexical features from a spaCy Doc object.

    Features include counts (characters, words, punctuation, quotes, all caps words,
    1st/2nd person pronouns) and ratios (average word length, punctuation ratio,
    all caps ratio).

    Args:
        doc: The spaCy Doc object representing the headline.

    Returns:
        A dictionary mapping feature names (e.g., 'feat_char_count') to their
        calculated values. Returns default zero values if the document contains
        no non-space tokens.
    """
    features: Dict[str, Union[int, float]] = {}
    non_space_tokens = [token for token in doc if not token.is_space]
    num_tokens = len(non_space_tokens)

    # Default values for empty or space-only headlines
    if num_tokens == 0:
        return {
            'feat_char_count': 0, 'feat_word_count': 0, 'feat_avg_word_length': 0.0,
            'feat_question_mark_flag': 0, 'feat_exclamation_mark_count': 0,
            'feat_quote_count': 0, 'feat_punct_count': 0, 'feat_punct_ratio': 0.0,
            'feat_all_caps_count': 0, 'feat_all_caps_ratio': 0.0,
            'feat_first_person_pron_count': 0, 'feat_second_person_pron_count': 0
        }

    # Calculate features
    features['feat_char_count'] = len(doc.text)
    features['feat_word_count'] = num_tokens
    features['feat_avg_word_length'] = np.mean([len(t.text) for t in non_space_tokens])
    features['feat_question_mark_flag'] = 1 if '?' in doc.text else 0
    features['feat_exclamation_mark_count'] = doc.text.count('!')
    features['feat_quote_count'] = sum(1 for token in doc if token.is_quote)
    features['feat_punct_count'] = sum(1 for token in doc if token.is_punct)
    features['feat_punct_ratio'] = features['feat_punct_count'] / num_tokens

    # Count words in all caps (longer than 1 character)
    all_caps_count = sum(1 for token in non_space_tokens if token.text.isupper() and len(token.text) > 1)
    features['feat_all_caps_count'] = all_caps_count
    features['feat_all_caps_ratio'] = all_caps_count / num_tokens

    # Count 1st and 2nd person pronouns (using lemmas)
    first_person_pronouns = {'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'}
    second_person_pronouns = {'you', 'your', 'yours'}
    features['feat_first_person_pron_count'] = sum(1 for token in doc if token.lemma_.lower() in first_person_pronouns)
    features['feat_second_person_pron_count'] = sum(1 for token in doc if token.lemma_.lower() in second_person_pronouns)

    return features


def get_pos_features(doc: spacy.tokens.doc.Doc) -> Dict[str, float]:
    """
    Extracts Part-of-Speech (POS) ratio features from a spaCy Doc object.

    Calculates the ratio of common POS tags (NOUN, VERB, ADJ, ADV, PROPN, PRON, DET)
    relative to the total number of non-punctuation and non-space tokens.

    Args:
        doc: The spaCy Doc object representing the headline.

    Returns:
        A dictionary mapping POS ratio feature names (e.g., 'feat_noun_ratio')
        to their calculated float values. Returns 0.0 for all ratios if no valid
        tokens are found.
    """
    # Count POS tags excluding punctuation and spaces
    pos_counts = Counter(token.pos_ for token in doc if not token.is_punct and not token.is_space)
    num_valid_tokens = sum(pos_counts.values())

    # Handle case with no valid tokens
    if num_valid_tokens == 0:
        return {
            'feat_noun_ratio': 0.0, 'feat_verb_ratio': 0.0, 'feat_adj_ratio': 0.0,
            'feat_adv_ratio': 0.0, 'feat_propn_ratio': 0.0, 'feat_pron_ratio': 0.0,
            'feat_det_ratio': 0.0
        }

    # Calculate ratios
    features: Dict[str, float] = {}
    target_pos = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'PRON', 'DET']
    for pos in target_pos:
        features[f'feat_{pos.lower()}_ratio'] = pos_counts.get(pos, 0) / num_valid_tokens

    return features


def get_sentiment_emotion_features(headline_text: str, doc: spacy.tokens.doc.Doc, sentiment_threshold: float = 0.2) -> Dict[str, Union[int, float]]:
    """
    Extracts sentiment (VADER) and emotion (NRCLex) features from headline text.

    Features:
    - Overall VADER scores (compound, positive, negative, neutral).
    - Sentence-level VADER analysis:
        - Variance of sentence compound scores.
        - Range (max - min) of sentence compound scores.
        - Oscillation flag (presence of both positive peak and negative trough).
        - Count of positive peaks (sentences > threshold).
        - Count of negative troughs (sentences < -threshold).
        - Maximum absolute deviation of any sentence score from the overall compound score.
        - Variance of these deviations.
        - Peak prominence (difference between max and 2nd max sentence score).
        - Trough prominence (difference between min and 2nd min sentence score).
    - NRCLex emotion frequencies (fear, anger, anticipation, trust, surprise,
      sadness, disgust, joy).

    Args:
        headline_text: The raw headline string.
        doc: The spaCy Doc object representing the headline.
        sentiment_threshold: The absolute compound score threshold used to define
                             a sentiment peak or trough in sentence analysis.

    Returns:
        A dictionary mapping sentiment and emotion feature names to their values.
        Returns 0.0 for NRCLex frequencies if the library fails or finds no affect words.
    """
    features: Dict[str, Union[int, float]] = {}
    if vader_analyzer is None:
        raise RuntimeError("VADER analyzer not initialized. Call setup_nlp_resources() first.")

    # --- Overall VADER Sentiment ---
    vader_scores = vader_analyzer.polarity_scores(headline_text)
    overall_compound = vader_scores['compound']
    features['feat_sentiment_vader_compound'] = overall_compound
    features['feat_sentiment_vader_pos'] = vader_scores['pos']
    features['feat_sentiment_vader_neg'] = vader_scores['neg']
    features['feat_sentiment_vader_neu'] = vader_scores['neu']

    # --- Sentence-Level VADER Analysis ---
    # Calculate compound score for each sentence
    sentence_sentiments = [vader_analyzer.polarity_scores(sent.text)['compound'] for sent in doc.sents]
    num_sentences = len(sentence_sentiments)

    # Initialize sentence-level features
    features.update({
        'feat_sentiment_variance': 0.0, 'feat_sentiment_range': 0.0,
        'feat_sentiment_oscillation_flag': 0, 'feat_sentiment_pos_peak_count': 0,
        'feat_sentiment_neg_trough_count': 0, 'feat_sentiment_max_deviation_from_overall': 0.0,
        'feat_sentiment_deviation_variance': 0.0, 'feat_sentiment_peak_prominence': 0.0,
        'feat_sentiment_trough_prominence': 0.0
    })

    if num_sentences > 0:
        sent_array = np.array(sentence_sentiments)
        min_sent, max_sent = np.min(sent_array), np.max(sent_array)
        features['feat_sentiment_range'] = max_sent - min_sent

        # Deviation from overall score
        deviations = np.abs(sent_array - overall_compound)
        if deviations.size > 0:
             features['feat_sentiment_max_deviation_from_overall'] = np.max(deviations)

        # Peaks and Troughs
        pos_peaks_mask = sent_array > sentiment_threshold
        neg_troughs_mask = sent_array < -sentiment_threshold
        features['feat_sentiment_pos_peak_count'] = np.sum(pos_peaks_mask)
        features['feat_sentiment_neg_trough_count'] = np.sum(neg_troughs_mask)

        # Features requiring more than one sentence
        if num_sentences > 1:
            features['feat_sentiment_variance'] = np.var(sent_array)
            if deviations.size > 0:
                 features['feat_sentiment_deviation_variance'] = np.var(deviations)

            # Oscillation: Check for presence of both a peak and a trough
            if np.any(pos_peaks_mask) and np.any(neg_troughs_mask):
                features['feat_sentiment_oscillation_flag'] = 1

            # Prominence: Difference between top 2 and bottom 2 scores
            sorted_sentiments = np.sort(sent_array)
            features['feat_sentiment_peak_prominence'] = sorted_sentiments[-1] - sorted_sentiments[-2]
            features['feat_sentiment_trough_prominence'] = sorted_sentiments[0] - sorted_sentiments[1]
        # Single sentence case: variance, prominence, oscillation remain 0

    # --- NRCLex Emotion Features ---
    emotions = ['fear', 'anger', 'anticipation', 'trust', 'surprise', 'sadness', 'disgust', 'joy']
    for emotion in emotions:
        features[f'feat_emotion_{emotion}_freq'] = 0.0

    try:
        nrc = NRCLex(headline_text)
        # Use affect_frequencies which returns scores for the core emotions
        emotion_freqs = nrc.affect_frequencies
        # Extract scores only for the desired core emotions
        emotion_scores = {e: emotion_freqs.get(e, 0.0) for e in emotions}
        total_affect_words = sum(emotion_scores.values())

        # Calculate frequencies if affect words were found
        if total_affect_words > 0:
            for emotion in emotions:
                features[f'feat_emotion_{emotion}_freq'] = emotion_scores[emotion] / total_affect_words
    except Exception as e:
        # Log warning optionally, but keep default 0 frequencies
        # print(f"Warning: NRCLex processing failed for '{headline_text[:50]}...': {e}")
        pass

    return features


def get_exaggeration_features(doc: spacy.tokens.doc.Doc, absurd_num_threshold: int = 1_000_000) -> Dict[str, Union[int, float]]:
    """
    Extracts features potentially indicative of exaggeration or hyperbole.

    Features include counts and ratios of intensifiers (e.g., 'very', 'extremely'),
    presence and magnitude of numbers (flagging potentially absurdly large ones),
    and counts of superlative and comparative adjectives/adverbs.

    Args:
        doc: The spaCy Doc object representing the headline.
        absurd_num_threshold: The numerical value above which a number found in the
                              text is flagged as potentially absurd ('feat_absurd_number_flag').

    Returns:
        A dictionary mapping exaggeration feature names to their calculated values.
    """
    features: Dict[str, Union[int, float]] = {
        'feat_intensifier_count': 0, 'feat_number_count': 0, 'feat_superlative_count': 0,
        'feat_comparative_count': 0, 'feat_intensifier_flag': 0, 'feat_contains_number_flag': 0,
        'feat_absurd_number_flag': 0, 'feat_intensifier_ratio': 0.0
    }
    non_space_tokens = [token for token in doc if not token.is_space]
    num_tokens = len(non_space_tokens)
    if num_tokens == 0:
        return features

    max_number_val = 0.0 # Track maximum absolute numerical value found

    for token in doc:
        # Check for intensifiers (case-insensitive lemma matching)
        if token.lemma_.lower() in INTENSIFIER_KEYWORDS:
            features['feat_intensifier_count'] += 1

        # Check for numbers
        if token.like_num:
            features['feat_number_count'] += 1
            try:
                # Attempt to parse the number, removing common symbols
                num_text = token.text.replace(',', '').replace('$', '').replace('%', '')
                current_num_val = float(num_text)
                max_number_val = max(abs(current_num_val), max_number_val)
            except ValueError:
                pass # Ignore if token looks like a number but isn't parseable

        # Check for superlatives (JJS, RBS tags) and comparatives (JJR, RBR tags)
        if token.tag_ in ['JJS', 'RBS']:
            features['feat_superlative_count'] += 1
        elif token.tag_ in ['JJR', 'RBR']:
            features['feat_comparative_count'] += 1

    # Set flags based on counts
    if features['feat_intensifier_count'] > 0:
        features['feat_intensifier_flag'] = 1
    if features['feat_number_count'] > 0:
        features['feat_contains_number_flag'] = 1
        # Flag if the largest number found exceeds the threshold
        if max_number_val >= absurd_num_threshold:
            features['feat_absurd_number_flag'] = 1

    # Calculate intensifier ratio
    features['feat_intensifier_ratio'] = features['feat_intensifier_count'] / num_tokens

    return features


def get_mundanity_ner_features(headline_text: str, doc: spacy.tokens.doc.Doc) -> Dict[str, Union[int, float]]:
    """
    Extracts features related to mundanity and Named Entity Recognition (NER).

    Mundanity features identify the use of generic person terms (e.g., "area man")
    and common, everyday action verbs (e.g., "uses", "enters").
    NER features count the occurrences of different entity types (PERSON, ORG, GPE, etc.)
    and calculate the average character length of detected entities.

    Args:
        headline_text: The raw headline string (used for regex matching of phrases).
        doc: The spaCy Doc object representing the headline.

    Returns:
        A dictionary mapping mundanity and NER feature names to their values.
    """
    features: Dict[str, Union[int, float]] = {
        'feat_generic_person_term_count': 0, 'feat_mundane_action_verb_count': 0,
        'feat_generic_person_term_flag': 0, 'feat_mundane_action_verb_flag': 0,
        'feat_mundane_combo_flag': 0, 'feat_ner_total_count': 0, 'feat_ner_person_count': 0,
        'feat_ner_org_count': 0, 'feat_ner_gpe_count': 0, 'feat_ner_norp_count': 0,
        'feat_ner_fac_count': 0, 'feat_ner_loc_count': 0, 'feat_ner_product_count': 0,
        'feat_ner_event_count': 0, 'feat_ner_avg_char_length': 0.0
    }

    # --- Mundanity Features ---
    generic_person_count = 0
    found_generic_person = False

    # Check single word generic terms using lemmas
    for token in doc:
        if token.lemma_.lower() in GENERIC_PERSON_SINGLE_WORDS:
            generic_person_count += 1
            found_generic_person = True

    # Check multi-word generic phrases using compiled regex on original text
    for pattern in GENERIC_PERSON_REGEX_COMPILED:
        matches = pattern.findall(headline_text)
        if matches:
            generic_person_count += len(matches)
            found_generic_person = True

    features['feat_generic_person_term_count'] = generic_person_count
    if found_generic_person:
        features['feat_generic_person_term_flag'] = 1

    # Check for mundane action verbs (lemma matching)
    mundane_verb_count = 0
    found_mundane_verb = False
    for token in doc:
        if token.lemma_ in MUNDANE_ACTION_VERBS and token.pos_ == 'VERB':
            mundane_verb_count += 1
            found_mundane_verb = True

    features['feat_mundane_action_verb_count'] = mundane_verb_count
    if found_mundane_verb:
        features['feat_mundane_action_verb_flag'] = 1

    # Flag if both mundane elements are present
    if features['feat_generic_person_term_flag'] == 1 and features['feat_mundane_action_verb_flag'] == 1:
        features['feat_mundane_combo_flag'] = 1

    # --- NER Features ---
    entities = doc.ents
    features['feat_ner_total_count'] = len(entities)

    if entities: # Only process if entities were found
        entity_labels = Counter(ent.label_ for ent in entities)
        features['feat_ner_person_count'] = entity_labels.get('PERSON', 0)
        features['feat_ner_org_count'] = entity_labels.get('ORG', 0)
        features['feat_ner_gpe_count'] = entity_labels.get('GPE', 0) # Geo-Political Entity
        features['feat_ner_norp_count'] = entity_labels.get('NORP', 0) # Nationality, Religious, Political
        features['feat_ner_fac_count'] = entity_labels.get('FAC', 0)  # Facility
        features['feat_ner_loc_count'] = entity_labels.get('LOC', 0)  # Location (non-GPE)
        features['feat_ner_product_count'] = entity_labels.get('PRODUCT', 0)
        features['feat_ner_event_count'] = entity_labels.get('EVENT', 0)

        # Calculate average length of entity text
        entity_lengths = [len(ent.text) for ent in entities]
        if entity_lengths: # Avoid division by zero
             features['feat_ner_avg_char_length'] = np.mean(entity_lengths)

    return features


def get_structure_incongruity_features(doc: spacy.tokens.doc.Doc, distance_threshold: float = 0.75) -> Dict[str, Union[int, float]]:
    """
    Extracts structural and semantic incongruity features using noun chunks and dependency parsing.

    Features include:
    - Noun Chunk Semantic Distance: Max and average cosine distance between noun chunk
      vectors, and a flag indicating if the max distance exceeds a threshold.
    - Dependency Features: Max depth of the dependency tree, average distance between
      a token and its head, counts and ratios of key dependency relations (nsubj, dobj,
      amod, advcl, pobj).

    Args:
        doc: The spaCy Doc object representing the headline.
        distance_threshold: The cosine distance value above which the semantic distance
                          between noun chunks is considered high ('feat_nc_semantic_distance_gt_threshold_flag').

    Returns:
        A dictionary mapping structural and incongruity feature names to their values.
    """
    features: Dict[str, Union[int, float]] = {
        'feat_nc_max_semantic_distance': 0.0, 'feat_nc_avg_semantic_distance': 0.0,
        'feat_nc_semantic_distance_gt_threshold_flag': 0, 'feat_dep_max_tree_depth': 0,
        'feat_dep_avg_distance': 0.0, 'feat_dep_nsubj_count': 0, 'feat_dep_dobj_count': 0,
        'feat_dep_amod_count': 0, 'feat_dep_advcl_count': 0, 'feat_dep_pobj_count': 0,
        'feat_dep_nsubj_ratio': 0.0, 'feat_dep_dobj_ratio': 0.0
    }

    # --- Noun Chunk Semantic Distance ---
    # Filter noun chunks to include only those with valid vectors
    noun_chunks = [chunk for chunk in doc.noun_chunks if chunk.has_vector and np.any(chunk.vector)]

    if len(noun_chunks) >= 2: # Need at least two chunks to compare
        distances = []
        for i in range(len(noun_chunks)):
            for j in range(i + 1, len(noun_chunks)):
                vec1 = noun_chunks[i].vector
                vec2 = noun_chunks[j].vector
                distance = cosine(vec1, vec2) # Cosine distance = 1 - similarity
                distances.append(np.clip(distance, 0.0, 2.0)) # Ensure distance is within [0, 2]

        if distances: # Ensure distances were calculated
            max_dist = max(distances)
            features['feat_nc_max_semantic_distance'] = max_dist
            features['feat_nc_avg_semantic_distance'] = np.mean(distances)
            if max_dist > distance_threshold:
                features['feat_nc_semantic_distance_gt_threshold_flag'] = 1

    # --- Dependency Features ---
    max_depth = 0
    dep_distances = [] # Store distances between tokens and their heads
    dep_counts = Counter() # Count types of dependency relations
    num_valid_tokens_for_ratio = 0 # Count non-punct/space tokens for normalization

    for token in doc:
        is_valid_token = not token.is_punct and not token.is_space

        if is_valid_token:
            num_valid_tokens_for_ratio += 1

            # Calculate max dependency tree depth for this token
            current_depth = 0
            current = token
            while current.head != current: # Traverse up to the root
                current_depth += 1
                current = current.head
            max_depth = max(current_depth, max_depth)

            # Calculate distance to head (if not root)
            if token.dep_ != 'ROOT':
                dep_distances.append(abs(token.i - token.head.i))

        # Count dependency relation type for all tokens
        dep_counts[token.dep_] += 1

    features['feat_dep_max_tree_depth'] = max_depth
    if dep_distances: # Calculate average distance if list is not empty
        features['feat_dep_avg_distance'] = np.mean(dep_distances)

    # Aggregate specific dependency counts
    features['feat_dep_nsubj_count'] = dep_counts.get('nsubj', 0) + dep_counts.get('nsubjpass', 0) # Nominal subject (active+passive)
    features['feat_dep_dobj_count'] = dep_counts.get('dobj', 0)    # Direct object
    features['feat_dep_amod_count'] = dep_counts.get('amod', 0)    # Adjectival modifier
    features['feat_dep_advcl_count'] = dep_counts.get('advcl', 0)   # Adverbial clause modifier
    features['feat_dep_pobj_count'] = dep_counts.get('pobj', 0)    # Object of preposition

    # Calculate ratios based on the number of valid (non-punct/space) tokens
    if num_valid_tokens_for_ratio > 0:
        features['feat_dep_nsubj_ratio'] = features['feat_dep_nsubj_count'] / num_valid_tokens_for_ratio
        features['feat_dep_dobj_ratio'] = features['feat_dep_dobj_count'] / num_valid_tokens_for_ratio
        # Ratios for other counts (amod, advcl, pobj) could be added similarly

    return features


def get_interaction_features(features_dict: Dict[str, Union[int, float]]) -> Dict[str, float]:
    """
    Generates interaction features by combining existing base features, using magnitudes.

    Creates new features by multiplying or dividing pairs or groups of base features
    to capture potential combined effects related to sarcasm (e.g., mundanity combined
    with exaggeration, sentiment/emotion contradictions, incongruity and structure).

    Args:
        features_dict: Dictionary containing the pre-calculated base features for a headline.

    Returns:
        A dictionary mapping interaction feature names (e.g.,
        'feat_interact_vader_neg_x_pos_emotion_freq') to their calculated float values.
    """
    interactions: Dict[str, float] = {}
    default_zero = 0.0
    epsilon = 1e-6 # Small constant to prevent division by zero

    # --- Mundanity x Exaggeration ---
    # Mundane context flag * intensifier ratio
    interactions['feat_interact_mundane_combo_x_intensifier_ratio'] = (
        features_dict.get('feat_mundane_combo_flag', 0) *
        features_dict.get('feat_intensifier_ratio', default_zero)
    )
    # Generic person flag * count of superlatives
    interactions['feat_interact_generic_person_flag_x_superlative_count'] = (
        features_dict.get('feat_generic_person_term_flag', 0) *
        features_dict.get('feat_superlative_count', default_zero)
    )

    # --- Sentiment/Emotion Contradictions/Interactions ---
    # Sum of positive emotion frequencies (Joy, Trust, Anticipation)
    pos_emotion_freq_sum = sum(features_dict.get(f'feat_emotion_{e}_freq', default_zero)
                               for e in ['joy', 'trust', 'anticipation'])
    # VADER negative score * sum of positive emotion frequencies
    interactions['feat_interact_vader_neg_x_pos_emotion_freq'] = (
        features_dict.get('feat_sentiment_vader_neg', default_zero) * pos_emotion_freq_sum
    )

    # Sum of negative emotion frequencies (Anger, Fear, Sadness, Disgust)
    neg_emotion_freq_sum = sum(features_dict.get(f'feat_emotion_{e}_freq', default_zero)
                               for e in ['anger', 'fear', 'sadness', 'disgust'])
    # VADER positive score * sum of negative emotion frequencies
    interactions['feat_interact_vader_pos_x_neg_emotion_freq'] = (
        features_dict.get('feat_sentiment_vader_pos', default_zero) * neg_emotion_freq_sum
    )

    # Sentiment range * intensifier ratio
    interactions['feat_interact_sentiment_range_x_intensifier_ratio'] = (
        features_dict.get('feat_sentiment_range', default_zero) *
        features_dict.get('feat_intensifier_ratio', default_zero)
    )
    # Sentiment oscillation flag * surprise frequency
    interactions['feat_interact_oscillation_flag_x_surprise_freq'] = (
        features_dict.get('feat_sentiment_oscillation_flag', 0) *
        features_dict.get('feat_emotion_surprise_freq', default_zero)
    )

    # --- Incongruity x Specificity/Structure ---
    # Max noun chunk distance * average NER length
    interactions['feat_interact_nc_max_distance_x_ner_avg_len'] = (
        features_dict.get('feat_nc_max_semantic_distance', default_zero) *
        features_dict.get('feat_ner_avg_char_length', default_zero)
    )
    # Max noun chunk distance * max dependency depth
    interactions['feat_interact_nc_max_distance_x_dep_max_depth'] = (
        features_dict.get('feat_nc_max_semantic_distance', default_zero) *
        features_dict.get('feat_dep_max_tree_depth', default_zero)
    )

    # --- Additive/Ratio Interactions ---
    # Ratio of positive to negative emotion frequency sums
    interactions['feat_interact_pos_neg_emotion_ratio'] = (
        pos_emotion_freq_sum / (neg_emotion_freq_sum + epsilon)
    )
    # Overall VADER compound score * sentiment range
    interactions['feat_interact_vader_compound_x_sentiment_range'] = (
        features_dict.get('feat_sentiment_vader_compound', default_zero) *
        features_dict.get('feat_sentiment_range', default_zero)
    )

    return interactions


def extract_all_optimized_features(headline: str) -> Optional[Dict[str, Union[int, float]]]:
    """
    Master function to extract the full suite of features for a single headline.

    Orchestrates calls to individual feature extraction functions in sequence,
    passing the necessary arguments (headline text, spaCy Doc). The interaction
    features are calculated last based on the dictionary of previously extracted features.

    Requires global `nlp` (spaCy model) to be initialized via `setup_nlp_resources()`.

    Args:
        headline: The headline text string.

    Returns:
        A dictionary containing all extracted features (base + interaction),
        or None if the input headline is invalid or an unhandled error occurs
        during processing.
    """
    # Basic input validation
    if not headline or not isinstance(headline, str) or not headline.strip():
        print(f"Warning: Skipping invalid headline input: {headline}")
        return None
    if nlp is None:
        raise RuntimeError("spaCy model 'nlp' not initialized. Call setup_nlp_resources() first.")

    try:
        # Process headline with spaCy once
        doc = nlp(headline)
        all_features: Dict[str, Union[int, float]] = {}

        # Call base feature extraction functions
        all_features.update(get_lexical_basic_features(doc))
        all_features.update(get_pos_features(doc))
        all_features.update(get_sentiment_emotion_features(headline, doc))
        all_features.update(get_exaggeration_features(doc))
        all_features.update(get_mundanity_ner_features(headline, doc))
        all_features.update(get_structure_incongruity_features(doc))

        # Calculate interaction features based on the collected base features
        all_features.update(get_interaction_features(all_features))

        return all_features

    except Exception as e:
        # Log error and return None for robustness
        print(f"ERROR during feature extraction for headline: '{headline[:100]}...' - Error: {e}")
        # Consider logging the full traceback here if needed for debugging
        # import traceback
        # traceback.print_exc()
        return None


# --- 3. Data Loading and Processing ---

def load_raw_data(raw_dataset_path: str) -> pd.DataFrame:
    """
    Loads raw dataset from a specified CSV or JSON/JSONL file path.

    Validates file existence and format, and checks for required columns
    ('is_sarcastic', 'headline').

    Args:
        raw_dataset_path: The file path to the raw dataset.

    Returns:
        A pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the file format is unsupported or required columns are missing.
        Exception: For other potential loading errors (e.g., malformed files).
    """
    print(f"--- Loading Raw Data from: {raw_dataset_path} ---")
    if not os.path.exists(raw_dataset_path):
         raise FileNotFoundError(f"Dataset file not found at {raw_dataset_path}")

    try:
        # Load based on file extension
        if raw_dataset_path.lower().endswith('.csv'):
            df = pd.read_csv(raw_dataset_path)
        elif raw_dataset_path.lower().endswith(('.json', '.jsonl')):
            try: # Attempt reading as JSON Lines first
                df = pd.read_json(raw_dataset_path, lines=True)
            except ValueError: # Fallback to standard JSON array
                print("Reading as JSON lines failed, trying standard JSON array format...")
                df = pd.read_json(raw_dataset_path)
        else:
            raise ValueError("Unsupported file format. Please use .csv or .json/.jsonl")

        # Validate required columns
        required_cols = ['is_sarcastic', 'headline']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Dataset must contain columns: {required_cols}. Found: {df.columns.tolist()}")

        print(f"Raw dataset loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"ERROR loading raw dataset: {e}")
        raise # Re-raise error after logging


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic cleaning on the raw DataFrame.

    Steps:
    1. Drops rows with missing 'headline' or 'is_sarcastic' values.
    2. Converts 'headline' column to string type and strips leading/trailing whitespace.
    3. Converts 'is_sarcastic' column to integer type, raising error if not possible.
    4. Removes rows where the 'headline' became empty after stripping whitespace.

    Args:
        df: The raw pandas DataFrame, expected to have 'headline' and 'is_sarcastic'.

    Returns:
        A cleaned pandas DataFrame.

    Raises:
        ValueError: If 'is_sarcastic' cannot be converted to int, or if no valid
                    rows remain after cleaning.
    """
    print("--- Cleaning Data ---")
    initial_rows = len(df)

    # Drop rows with missing critical values
    df.dropna(subset=['headline', 'is_sarcastic'], inplace=True)

    # Clean 'headline' column
    df['headline'] = df['headline'].astype(str).str.strip()

    # Ensure 'is_sarcastic' is integer
    try:
        df['is_sarcastic'] = df['is_sarcastic'].astype(int)
        # Optional: Check if target is binary (0 or 1)
        if not df['is_sarcastic'].isin([0, 1]).all():
             warnings.warn("Target variable 'is_sarcastic' contains values other than 0 or 1.", UserWarning)
    except ValueError:
        raise ValueError("Column 'is_sarcastic' contains non-integer values and could not be converted.")

    # Remove rows with empty headlines after cleaning
    df = df[df['headline'].str.len() > 0]
    rows_after_cleaning = len(df)
    print(f"Rows remaining after cleaning: {rows_after_cleaning} ({initial_rows - rows_after_cleaning} removed)")

    if rows_after_cleaning == 0:
        raise ValueError("No valid headlines/targets remaining after cleaning process.")
    return df


def extract_features_and_impute(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts the full feature set for all headlines in the DataFrame and imputes missing values.

    Iterates through the 'headline' column, applies `extract_all_optimized_features`,
    and collects the results. Filters out any rows where feature extraction failed (returned None).
    Then, imputes any remaining missing feature values (NaN or Inf) using the median strategy.

    Args:
        df: The cleaned pandas DataFrame containing 'headline' and 'is_sarcastic' columns.

    Returns:
        A tuple containing:
            - X (pd.DataFrame): DataFrame of imputed features.
            - y (pd.Series): Corresponding target labels for the rows where feature
                             extraction was successful.

    Raises:
        ValueError: If feature extraction fails for all headlines or if a mismatch occurs
                    between the number of feature rows and target labels after filtering.
    """
    print("--- Extracting Features and Imputing ---")
    start_time = time.time()

    # Apply feature extraction to the 'headline' column
    if TQDM_AVAILABLE:
        tqdm.pandas(desc="Extracting Features")
        feature_dicts_series = df['headline'].progress_apply(extract_all_optimized_features)
    else:
        print("Applying feature extraction (tqdm progress bar not available)...")
        feature_dicts_series = df['headline'].apply(extract_all_optimized_features)

    # Separate successful results from failures (None)
    valid_indices = feature_dicts_series.dropna().index
    feature_list = feature_dicts_series.loc[valid_indices].tolist()

    # Filter the original DataFrame and target to match successful extractions
    df_filtered = df.loc[valid_indices].reset_index(drop=True)

    if not feature_list:
        raise ValueError("Feature extraction returned no valid results. Check feature functions or input data.")

    features_df = pd.DataFrame(feature_list)
    y = df_filtered['is_sarcastic'] # Corresponding target labels

    # Verify consistency between features and target after filtering
    if len(features_df) != len(y):
         raise ValueError(f"Feature rows ({len(features_df)}) and target rows ({len(y)}) count mismatch after filtering failed extractions.")

    end_time = time.time()
    print(f"Feature extraction completed in {end_time - start_time:.2f} seconds.")
    print(f"Feature matrix shape before imputation: {features_df.shape}")

    # --- Imputation ---
    print("Performing median imputation on extracted features...")
    # Check for NaNs/Infs *before* imputation
    nan_count_before = features_df.isnull().sum().sum()
    # Use .values to check entire numpy array for inf
    inf_count_before = np.isinf(features_df.values).sum()
    print(f"NaN values before imputation: {nan_count_before}")
    print(f"Infinite values before imputation: {inf_count_before}")

    # Replace infinities with NaN so SimpleImputer can handle them
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute missing values using the median (robust to outliers)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(features_df)

    # Convert imputed array back to DataFrame with original column names
    X = pd.DataFrame(X_imputed, columns=features_df.columns)

    # Verify imputation results
    nan_count_after = X.isnull().sum().sum()
    inf_count_after = np.isinf(X.values).sum()
    print(f"NaN values after imputation: {nan_count_after}")
    print(f"Infinite values after imputation: {inf_count_after}")
    if nan_count_after > 0 or inf_count_after > 0:
        print("WARNING: NaNs or Infs still present after imputation! Check feature generation.")
        # Optionally, list columns with remaining issues:
        # print("Columns with NaNs:", X.columns[X.isnull().any()].tolist())
        # print("Columns with Infs:", X.columns[np.isinf(X.values).any(axis=0)].tolist())

    return X, y


def save_features_target(X: pd.DataFrame, y: pd.Series, x_path: str, y_path: str) -> None:
    """
    Saves the feature matrix (X) and target vector (y) to separate CSV files.

    Args:
        X: The pandas DataFrame of features.
        y: The pandas Series of target labels.
        x_path: File path where the features CSV should be saved.
        y_path: File path where the target CSV should be saved.

    Raises:
        IOError: If writing to either CSV file fails.
    """
    print(f"--- Saving Features and Target ---")
    try:
        print(f"Saving features (X) to {x_path}")
        X.to_csv(x_path, index=False) # index=False is important
        print(f"Saving target (y) to {y_path}")
        # Save target Series as a DataFrame with a header for easier loading
        y.to_frame(name='is_sarcastic').to_csv(y_path, index=False)
        print("Features and target saved successfully.")
    except IOError as e:
        print(f"ERROR saving features/target to CSV: {e}")
        raise # Re-raise to signal failure


def load_features_target(x_path: str, y_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads pre-computed features (X) and target (y) from specified CSV files.

    Performs validation checks on file existence, target file format, and row count consistency.

    Args:
        x_path: File path to the features CSV file.
        y_path: File path to the target CSV file.

    Returns:
        A tuple containing the loaded features DataFrame (X) and target Series (y).

    Raises:
        FileNotFoundError: If either the features or target CSV file is not found.
        ValueError: If the loaded target CSV does not contain exactly one column,
                    or if the number of rows in X and y do not match.
        Exception: For other potential CSV loading errors.
    """
    print(f"--- Loading Pre-saved Features and Target ---")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Required feature/target CSV files not found at {x_path} or {y_path}")

    try:
        print(f"Loading features (X) from {x_path}")
        X = pd.read_csv(x_path)
        print(f"Loading target (y) from {y_path}")
        y_df = pd.read_csv(y_path)

        # Validate the structure of the loaded target file
        if len(y_df.columns) != 1:
            raise ValueError(f"Loaded target CSV '{y_path}' must have exactly one column, found {len(y_df.columns)}.")
        y = y_df[y_df.columns[0]] # Extract the target Series

        # Ensure consistency between features and target
        if len(X) != len(y):
            raise ValueError(f"Row count mismatch between loaded features ({len(X)}) and target ({len(y)}). Files may be corrupted or from different runs.")

        print("Pre-saved features and target loaded successfully.")
        print(f"Loaded features shape: {X.shape}, Loaded target shape: {y.shape}")
        return X, y
    except Exception as e:
        print(f"ERROR loading features/target from CSV: {e}")
        raise


def get_data(raw_dataset_path: str, force_extract: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Manages the data acquisition workflow: load pre-computed features or extract them.

    Attempts to load features and target from CSV files (FEATURES_X_CSV, TARGET_Y_CSV)
    if they exist and `force_extract` is False. If loading fails or `force_extract`
    is True, it initiates the full process: load raw data -> clean -> extract features
    -> impute -> save results to CSV.

    Args:
        raw_dataset_path: Path to the raw dataset file (CSV or JSON/JSONL).
        force_extract: If True, skips loading attempt and forces feature re-extraction.

    Returns:
        A tuple containing the final features DataFrame (X) and target Series (y).

    Raises:
        FileNotFoundError: If raw data file is not found during extraction process.
        ValueError: If cleaning or feature extraction leads to invalid states.
        Exception: For other critical errors during loading or processing.
    """
    print("--- Managing Data Acquisition ---")
    # Attempt to load pre-computed features unless forced to extract
    if not force_extract and os.path.exists(FEATURES_X_CSV) and os.path.exists(TARGET_Y_CSV):
        try:
            X, y = load_features_target(FEATURES_X_CSV, TARGET_Y_CSV)
            print("Using pre-saved features.")
            return X, y
        except (FileNotFoundError, ValueError, Exception) as e:
            print(f"Could not load pre-saved features ({e}). Proceeding with full extraction.")
            # Ensure flag is set for clarity, although logic proceeds anyway
            force_extract = True

    # --- Full Extraction Process ---
    print("Initiating full data loading and feature extraction...")
    try:
        df_raw = load_raw_data(raw_dataset_path)
        df_cleaned = clean_data(df_raw)
        X, y = extract_features_and_impute(df_cleaned)

        # Attempt to save the newly extracted features
        try:
            save_features_target(X, y, FEATURES_X_CSV, TARGET_Y_CSV)
        except IOError as e:
            # Log warning but don't halt execution if saving fails
            print(f"WARNING: Failed to save newly extracted features: {e}")

        print("Full data processing and feature extraction complete.")
        return X, y

    except (FileNotFoundError, ValueError, Exception) as e:
        # Catch critical errors during the extraction process
        print(f"FATAL ERROR during data processing/feature extraction: {e}")
        # Re-raise or exit depending on desired script behavior
        raise


# --- 4. Dimensionality Reduction (Optional) ---

def apply_pca(X_train_scaled: np.ndarray, X_test_scaled: np.ndarray, n_components: Union[int, float, None]) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Applies Principal Component Analysis (PCA) for dimensionality reduction.

    Fits PCA on the scaled training data and then transforms both the training
    and test sets using the fitted PCA model.

    Args:
        X_train_scaled: The scaled training feature matrix (numpy array).
        X_test_scaled: The scaled test feature matrix (numpy array).
        n_components: The number of principal components to keep.
                      - If int: Keep exactly this many components.
                      - If float (0 < n < 1): Keep the number of components that
                        explain at least this fraction of the total variance.
                      - If None: Keep all components (min(n_samples, n_features)).

    Returns:
        A tuple containing:
            - X_train_pca (np.ndarray): Transformed training data with reduced dimensions.
            - X_test_pca (np.ndarray): Transformed test data with reduced dimensions.
            - pca_model (PCA): The fitted scikit-learn PCA object.
    """
    print(f"\n--- Applying PCA (n_components={n_components}) ---")
    start_time = time.time()

    # Initialize and fit PCA on the training data ONLY
    pca_model = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca_model.fit_transform(X_train_scaled)

    # Transform the test data using the *same* fitted PCA model
    X_test_pca = pca_model.transform(X_test_scaled)

    end_time = time.time()
    print(f"PCA applied in {end_time - start_time:.2f} seconds.")
    print(f"Original number of features: {X_train_scaled.shape[1]}")
    print(f"Number of features after PCA: {pca_model.n_components_}")

    # Print explained variance if meaningful
    # Check if n_components allows for calculation of explained variance ratio sum
    if pca_model.n_components_ < X_train_scaled.shape[1]:
         print(f"Explained variance ratio by {pca_model.n_components_} components: {pca_model.explained_variance_ratio_.sum():.4f}")
    else:
        print("Explained variance ratio: 1.0 (all components retained)")


    return X_train_pca, X_test_pca, pca_model


# --- 5. Model Definitions ---

class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for binary classification using PyTorch.

    Architecture:
    Input -> Linear(input_dim, hidden_dim1) -> ReLU -> Dropout(dropout_rate) ->
    Linear(hidden_dim1, hidden_dim2) -> ReLU -> Dropout(dropout_rate) ->
    Linear(hidden_dim2, 2) -> Output (Logits for CrossEntropyLoss)
    """
    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout_rate: float = 0.3):
        """
        Initializes the MLP layers.

        Args:
            input_dim: The number of input features.
            hidden_dim1: The number of neurons in the first hidden layer. Defaults to 64.
            hidden_dim2: The number of neurons in the second hidden layer. Defaults to 32.
            dropout_rate: The dropout probability applied after each ReLU activation.
                          Defaults to 0.3.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 2) # 2 output units for binary classification
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the MLP.

        Args:
            x: The input tensor batch (shape: [batch_size, input_dim]).

        Returns:
            The output tensor (logits) (shape: [batch_size, 2]).
        """
        return self.network(x)


class PyTorchMLPWrapper(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible wrapper for the PyTorch SimpleMLP model.

    Enables using the PyTorch MLP within scikit-learn pipelines, model selection tools
    (like GridSearchCV, though less common for DL), and ensemble methods
    (like VotingClassifier, StackingClassifier). Handles the training loop,
    device management (CPU/GPU), prediction, and probability estimation.
    """
    def __init__(self, input_dim: Optional[int] = None, hidden_dim1: int = 64, hidden_dim2: int = 32,
                 dropout_rate: float = 0.3, epochs: int = 10, batch_size: int = 64,
                 lr: float = 0.001, random_state: Optional[int] = None, verbose: bool = False):
        """
        Initializes the PyTorchMLPWrapper.

        Args:
            input_dim: The number of input features. If None, it will be inferred
                       from the data during the first call to `fit`.
            hidden_dim1: Size of the first hidden layer. Defaults to 64.
            hidden_dim2: Size of the second hidden layer. Defaults to 32.
            dropout_rate: Dropout probability. Defaults to 0.3.
            epochs: Number of training epochs. Defaults to 10.
            batch_size: Training batch size. Defaults to 64.
            lr: Learning rate for the Adam optimizer. Defaults to 0.001.
            random_state: Seed for PyTorch's random number generator for reproducibility.
                          Defaults to None.
            verbose: If True, prints training progress (loss per epoch or periodically).
                     Defaults to False.
        """
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose
        self.model_: Optional[SimpleMLP] = None # Holds the actual PyTorch model instance
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes_ = np.array([0, 1]) # Necessary for scikit-learn compatibility
        self._is_fitted = False          # Tracks if fit has been called

    def _init_model(self, input_dim: int) -> None:
        """Initializes the internal PyTorch SimpleMLP model and moves it to the device."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                 torch.cuda.manual_seed_all(self.random_state) # For GPU reproducibility
        self.model_ = SimpleMLP(input_dim=input_dim, hidden_dim1=self.hidden_dim1,
                                hidden_dim2=self.hidden_dim2, dropout_rate=self.dropout_rate).to(self.device_)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'PyTorchMLPWrapper':
        """
        Fits the MLP model to the training data.

        Sets up the data loader, optimizer, loss function, and executes the
        training loop for the specified number of epochs. Infers `input_dim`
        if it wasn't provided during initialization. Sets the `_is_fitted` flag.

        Args:
            X: Training features (numpy array or pandas DataFrame).
            y: Training target labels (numpy array or pandas Series, expected values 0 or 1).

        Returns:
            The fitted wrapper instance `self`.

        Raises:
            ValueError: If `input_dim` was provided during init and does not match
                        the number of features in `X`.
        """
        # Check if already fitted - useful for ensemble methods that might clone
        if self._is_fitted:
            if self.verbose:
                print(f"INFO: PyTorch MLP Wrapper (RS={self.random_state}) is already fitted. Skipping re-training.")
            if not hasattr(self, 'n_features_in_'): # Ensure attribute exists if skipping
                self.n_features_in_ = X.shape[1]
            return self

        # Determine input dimension
        current_input_dim = X.shape[1]
        if self.input_dim is None:
            self.input_dim_ = current_input_dim # Infer if not set
        elif self.input_dim != current_input_dim:
            raise ValueError(f"Input dimension mismatch: wrapper initialized with input_dim={self.input_dim}, but fit data has {current_input_dim} features.")
        else:
            self.input_dim_ = self.input_dim # Use provided dimension

        self._init_model(self.input_dim_) # Initialize the actual PyTorch model

        # --- Data Preparation ---
        # Convert y to numpy array if it's a pandas Series
        y_np = y.values if isinstance(y, pd.Series) else np.asarray(y)
        # Convert X to numpy array if it's a pandas DataFrame
        X_np = X if isinstance(X, np.ndarray) else X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        # Convert data to PyTorch tensors and move to the appropriate device
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device_)
        y_tensor = torch.tensor(y_np, dtype=torch.long).to(self.device_) # Target needs to be Long for CrossEntropyLoss

        dataset = TensorDataset(X_tensor, y_tensor)
        # Consider drop_last=True if batch size issues arise, but usually False is fine
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        # --- Setup Optimizer and Loss ---
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        # --- Training Loop ---
        self.model_.train() # Set model to training mode
        if self.verbose:
            print(f"Training PyTorch MLP (RS={self.random_state}) on {self.device_} for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()        # Reset gradients per batch
                outputs = self.model_(batch_X)  # Forward pass
                loss = criterion(outputs, batch_y) # Compute loss
                loss.backward()              # Backpropagation
                optimizer.step()             # Update weights
                epoch_loss += loss.item()    # Accumulate batch loss
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            # Print progress: first epoch, last epoch, or every 10 epochs
            if self.verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch == self.epochs - 1):
                print(f"  Epoch [{epoch + 1}/{self.epochs}], Average Loss: {avg_epoch_loss:.4f}")

        if self.verbose:
            print(f"MLP (RS={self.random_state}) Training finished.")

        # Set scikit-learn required attributes after fitting
        self.n_features_in_ = self.input_dim_
        self._is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predicts class labels (0 or 1) for the input samples.

        Args:
            X: Input features (numpy array or pandas DataFrame).

        Returns:
            A numpy array of predicted class labels.

        Raises:
            RuntimeError: If the `fit` method has not been called yet.
        """
        check_is_fitted(self) # Verify model has been fitted
        self.model_.eval() # Set model to evaluation mode (disables dropout)

        # Prepare input tensor
        X_np = X if isinstance(X, np.ndarray) else X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device_)

        with torch.no_grad(): # Disable gradient computation for inference
            outputs = self.model_(X_tensor)
            # Get the index (0 or 1) corresponding to the highest output logit
            _, predicted_indices = torch.max(outputs, 1)

        # Move predictions to CPU and convert to numpy array
        return predicted_indices.cpu().numpy()

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predicts class probabilities for the input samples.

        Args:
            X: Input features (numpy array or pandas DataFrame).

        Returns:
            A numpy array of shape (n_samples, 2), where each row contains the
            probabilities for class 0 and class 1, respectively.

        Raises:
            RuntimeError: If the `fit` method has not been called yet.
        """
        check_is_fitted(self) # Verify model has been fitted
        self.model_.eval() # Set model to evaluation mode

        # Prepare input tensor
        X_np = X if isinstance(X, np.ndarray) else X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device_)

        with torch.no_grad(): # Disable gradient computation
            outputs = self.model_(X_tensor)
            # Apply softmax to convert output logits into probabilities
            probabilities = torch.softmax(outputs, dim=1)

        # Move probabilities to CPU and convert to numpy array
        return probabilities.cpu().numpy()

    # --- Scikit-learn Compatibility Methods ---
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Gets parameters for this estimator, supporting scikit-learn's cloning mechanism.

        Args:
            deep: If True, return parameters of sub-objects that are estimators. (Ignored here)

        Returns:
            Dictionary of parameter names mapped to their values.
        """
        return {
            "input_dim": self.input_dim, "hidden_dim1": self.hidden_dim1,
            "hidden_dim2": self.hidden_dim2, "dropout_rate": self.dropout_rate,
            "epochs": self.epochs, "batch_size": self.batch_size, "lr": self.lr,
            "random_state": self.random_state, "verbose": self.verbose
        }

    def set_params(self, **parameters: Any) -> 'PyTorchMLPWrapper':
        """
        Sets the parameters of this estimator, supporting scikit-learn's model tuning.

        Args:
            **parameters: Dictionary of parameter names and values to set.

        Returns:
            The estimator instance `self`.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        # Optionally reset fitted status if parameters impacting the model are changed
        # self._is_fitted = False # Let sklearn handle cloning and refitting as needed
        return self


# --- 6. Model Training and Evaluation Functions ---

def train_individual_models(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, pd.Series],
    n_features: int,
    include_decision_tree: bool = True # Flag to control DT inclusion
    ) -> Dict[str, BaseEstimator]:
    """
    Initializes and trains a set of individual classification models.

    Models trained by default: Logistic Regression, Decision Tree (optional),
    Random Forest, and one PyTorch MLP instance.

    Args:
        X_train: The training feature matrix (should be scaled or processed).
        y_train: The training target labels.
        n_features: The number of input features (required for MLP initialization).
        include_decision_tree: If True, trains and includes a Decision Tree model.

    Returns:
        A dictionary mapping model names (e.g., "Logistic Regression") to their
        trained scikit-learn or wrapper instances.
    """
    print("\n--- Training Individual Base Models ---")
    models: Dict[str, BaseEstimator] = {}

    # Logistic Regression
    print("Training Logistic Regression...")
    start_time = time.time()
    # Using liblinear solver suitable for this scale, C=1.0 default regularization
    lr_model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1000, C=1.0)
    models["Logistic Regression"] = lr_model.fit(X_train, y_train)
    print(f"-> LR trained in {time.time() - start_time:.2f}s")

    # Decision Tree (Optional)
    if include_decision_tree:
        print("Training Decision Tree...")
        start_time = time.time()
        # Parameters aim to prevent excessive overfitting
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=14, min_samples_leaf=10, min_samples_split=20)
        models["Decision Tree"] = dt_model.fit(X_train, y_train)
        print(f"-> DT trained in {time.time() - start_time:.2f}s")
    else:
        print("Skipping Decision Tree training.")

    # Random Forest
    print("Training Random Forest...")
    start_time = time.time()
    # Using more estimators and setting complexity limits
    rf_model = RandomForestClassifier(
        random_state=42, n_estimators=200, max_depth=20,
        min_samples_leaf=5, min_samples_split=10, n_jobs=-1 # Use available cores
    )
    models["Random Forest"] = rf_model.fit(X_train, y_train)
    print(f"-> RF trained in {time.time() - start_time:.2f}s")

    # PyTorch MLP (Single instance)
    print("Training Single PyTorch MLP...")
    start_time = time.time()
    # Example MLP parameters, potentially tune these
    mlp_params = dict(
        input_dim=n_features, epochs=60, batch_size=32, lr=0.0008,
        hidden_dim1=128, hidden_dim2=64, dropout_rate=0.25, random_state=42, verbose=False
    )
    mlp_model = PyTorchMLPWrapper(**mlp_params)
    models["PyTorch MLP"] = mlp_model.fit(X_train, y_train)
    print(f"-> MLP trained in {time.time() - start_time:.2f}s")

    print("Individual models training complete.")
    return models

# --- XGBoost Training Function ---
def train_xgboost_model(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, pd.Series],
    params: Optional[Dict[str, Any]] = None,
    use_eval_set: bool = True,
    early_stopping_rounds_val: int = 50,
    eval_metric: Union[str, List[str]] = 'logloss',
    verbose_eval: Union[bool, int] = False, # Allow int for periodic printing
    random_state: int = 42
) -> Optional[xgb.XGBClassifier]: # Return Optional in case of import error
    """
    Initializes, trains, and returns an XGBoost classifier.

    Optionally uses early stopping based on a validation split of the training data.
    Handles potential ImportError if xgboost is not installed.

    Args:
        X_train: Training feature matrix (should be scaled or processed).
        y_train: Training target labels.
        params: Optional dictionary of XGBoost parameters to override defaults
                (excluding objective, seed, use_label_encoder, n_estimators, eval_metric).
        use_eval_set: If True, creates a validation split for early stopping.
        early_stopping_rounds_val: Number of rounds with no improvement for early stopping.
        eval_metric: Evaluation metric(s) for early stopping / model evaluation.
        verbose_eval: Controls printing of eval metrics during training when using eval_set.
                      Can be bool or int (frequency).
        random_state: Random seed for reproducibility.

    Returns:
        A trained xgboost.XGBClassifier instance, or None if xgboost is not installed.
    """
    if not XGBOOST_AVAILABLE:
        print("INFO: XGBoost library not found. Skipping XGBoost training.")
        return None

    print("Training XGBoost Model...")
    start_time = time.time()

    # --- Initialization ---
    init_params = {
        'objective': 'binary:logistic',
        'n_estimators': 1000, # Max rounds, may stop early
        'seed': random_state,
        'eta': 0.1, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'gamma': 0, 'lambda': 1, 'alpha': 0,
    }
    if params:
        init_params.update(params)

    # Label encoder usually not needed for recent versions with int targets
    label_encoder_needed = False

    model = xgb.XGBClassifier(
        use_label_encoder=label_encoder_needed,
        eval_metric=eval_metric,
        **init_params
    )

    # --- Prepare for Fitting (Early Stopping) ---
    eval_set_data = None
    should_use_early_stopping = False
    X_fit, y_fit = X_train, y_train # Train on full data by default

    if use_eval_set:
        try:
            X_train_part, X_val, y_train_part, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=random_state, stratify=y_train
            )
            eval_set_data = [(X_val, y_val)]
            X_fit, y_fit = X_train_part, y_train_part # Use subset for training
            should_use_early_stopping = False
            print(f"-> Using validation set ({len(y_val)} samples) for early stopping.")
        except Exception as e:
            print(f"WARNING: Could not create stratified validation set for XGBoost ({e}). Training without early stopping.")

    # --- Explicit Fit Call ---
    try:
        if should_use_early_stopping and eval_set_data:
            print(f"-> Fitting with early stopping (rounds={early_stopping_rounds_val})...")
            model.fit(
                X_fit, y_fit,
                eval_set=eval_set_data,
                early_stopping_rounds=early_stopping_rounds_val,
                verbose=verbose_eval
            )
        else:
            print("-> Fitting without early stopping...")
            model.fit(X_fit, y_fit, verbose=verbose_eval) # verbose still controls general fit messages

        print(f"-> XGBoost trained in {time.time() - start_time:.2f}s")

        if should_use_early_stopping:
            try:
                actual_rounds = model.best_iteration if hasattr(model, 'best_iteration') else getattr(model, 'best_ntree_limit', '?')
                print(f"-> Actual boosting rounds used: {actual_rounds}")
            except AttributeError: pass # Ignore if attributes not found

        return model

    except Exception as e:
        print(f"ERROR during XGBoost fitting: {e}")
        return None # Return None on fitting error


# --- SVM Training Function ---
def train_svm_model(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, pd.Series],
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42
) -> Optional[SVC]: # Return Optional in case of error
    """
    Initializes, trains, and returns an SVM classifier, defaulting to RBF kernel.

    Requires scaled input features for optimal performance. Enables probability
    estimates for potential use in soft voting or stacking.

    Args:
        X_train: Training feature matrix (MUST be scaled).
        y_train: Training target labels.
        params: Optional dictionary of SVC parameters to override defaults
                (e.g., {'C': 10, 'gamma': 0.1}).
        random_state: Random seed for reproducibility.

    Returns:
        A trained sklearn.svm.SVC instance, or None if training fails.
    """
    print("Training SVM Model...")
    start_time = time.time()

    # Default parameters (RBF kernel is often effective)
    default_params = {
        'kernel': 'rbf',
        'C': 1.0,             # Regularization strength
        'gamma': 'scale',     # Kernel coefficient ('scale' adjusts based on features)
        'probability': True,  # Essential for predict_proba needed by soft voting/stacking
        'random_state': random_state
    }
    if params:
        default_params.update(params)

    model = SVC(**default_params)

    try:
        model.fit(X_train, y_train)
        print(f"-> SVM ({default_params.get('kernel', 'rbf')} kernel) trained in {time.time() - start_time:.2f}s")
        return model
    except Exception as e:
        print(f"ERROR during SVM fitting: {e}")
        return None # Return None on error


# --- Stacking Ensemble Training Function ---
def train_stacking_model(
    base_models: Dict[str, BaseEstimator],
    X_train: np.ndarray,
    y_train: Union[np.ndarray, pd.Series],
    meta_model: Optional[BaseEstimator] = None,
    cv: int = 5,
    stack_method: str = 'predict_proba', # Default to probabilities for classification
    n_jobs: int = 1
) -> Optional[StackingClassifier]: # Return Optional in case of error
    """
    Creates and trains a StackingClassifier ensemble using pre-trained base models.

    Trains a meta-model on the out-of-fold predictions generated by the base models
    using cross-validation.

    Args:
        base_models: Dictionary mapping names to PRE-TRAINED base model instances.
                     Models must have `predict_proba` method if stack_method='predict_proba'.
        X_train: Training feature matrix (scaled or processed).
        y_train: Training target labels.
        meta_model: The meta-model (Level 1 estimator). If None, defaults to
                    LogisticRegression with default parameters.
        cv: Number of cross-validation folds for generating base model predictions.
        stack_method: Method used to compute outputs of base models ('predict_proba',
                      'decision_function', 'predict'). Defaults to 'predict_proba'.
        n_jobs: Number of cores to use (-1 for all). Use 1 if base models (e.g., PyTorch)
                have issues with parallelization.

    Returns:
        A trained sklearn.ensemble.StackingClassifier instance, or None if training fails.
    """
    print("Training Stacking Ensemble Model...")
    start_time = time.time()

    # Use Logistic Regression as default meta-model if none provided
    if meta_model is None:
        meta_model = LogisticRegression(random_state=42, solver='liblinear')
        print("-> Using default LogisticRegression as meta-model.")

    # Prepare list of (name, model) tuples
    estimators = list(base_models.items())
    if not estimators:
        print("ERROR: No base models provided for StackingClassifier.")
        return None

    # Basic validation checks
    valid_estimators = []
    final_stack_method = stack_method
    for name, model in estimators:
        try:
            check_is_fitted(model) # Check if base models appear fitted
            # Check compatibility with stack_method
            if final_stack_method == 'predict_proba' and not hasattr(model, 'predict_proba'):
                print(f"WARNING: Base model '{name}' lacks predict_proba. Stacking method falling back to 'predict'.")
                final_stack_method = 'predict'
            elif final_stack_method == 'decision_function' and not hasattr(model, 'decision_function'):
                print(f"WARNING: Base model '{name}' lacks decision_function. Stacking method falling back to 'predict'.")
                final_stack_method = 'predict'
            valid_estimators.append((name, model))
        except Exception as fit_check_error:
            # More robust check than just NotFittedError
            print(f"WARNING: Base model '{name}' check failed ({fit_check_error}). It might not be properly fitted or incompatible.")
            # Optionally skip this model, but for now we'll include it and let StackingClassifier handle errors
            valid_estimators.append((name, model)) # Keep it for StackingClassifier to potentially fail later if truly unfitted

    if not valid_estimators:
        print("ERROR: No valid/fitted base models available for StackingClassifier.")
        return None

    print(f"-> Using stack_method: '{final_stack_method}'")

    # Initialize StackingClassifier
    stacking_clf = StackingClassifier(
        estimators=valid_estimators,
        final_estimator=meta_model,
        cv=cv,
        stack_method=final_stack_method,
        n_jobs=n_jobs,
        passthrough=False # Meta-model sees only base model predictions
    )

    try:
        # Fit the meta-model using cross-validated predictions of base models
        stacking_clf.fit(X_train, y_train)
        print(f"-> Stacking Ensemble trained in {time.time() - start_time:.2f}s")
        return stacking_clf
    except Exception as e:
        print(f"ERROR during Stacking Ensemble fitting: {e}")
        return None


# --- MLP Ensemble Training Function ---
def train_mlp_ensemble(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, pd.Series],
    n_features: int
    ) -> Tuple[Optional[VotingClassifier], Optional[VotingClassifier]]: # Return Optional
    """
    Initializes, trains, and combines three MLP models into soft and hard voting ensembles.

    Trains three separate `PyTorchMLPWrapper` instances with the same hyperparameters
    but different `random_state` values to encourage diversity through weight initialization.

    Args:
        X_train: The training feature matrix (should be scaled or processed).
        y_train: The training target labels.
        n_features: The number of input features for the MLPs.

    Returns:
        A tuple containing:
            - The fitted soft voting MLP ensemble (VotingClassifier or None if failed).
            - The fitted hard voting MLP ensemble (VotingClassifier or None if failed).
    """
    print("\n--- Training 3x MLP Ensemble ---")
    # Define base parameters for the MLPs in the ensemble
    mlp_params_base = dict(
        input_dim=n_features, epochs=60, batch_size=32, lr=0.0008,
        hidden_dim1=128, hidden_dim2=64, dropout_rate=0.25, verbose=False
    )
    mlp_seeds = [42, 123, 999] # Seeds for initialization diversity

    mlp_instances: List[PyTorchMLPWrapper] = []
    training_successful = True
    for i, seed in enumerate(mlp_seeds):
        print(f"Training PyTorch MLP Instance {i+1} (RS={seed})...")
        start_time = time.time()
        mlp = PyTorchMLPWrapper(**mlp_params_base, random_state=seed)
        try:
            mlp.fit(X_train, y_train)
            mlp_instances.append(mlp)
            print(f"-> MLP {i+1} trained in {time.time() - start_time:.2f}s")
        except Exception as e:
            print(f"ERROR training MLP Instance {i+1} (RS={seed}): {e}")
            training_successful = False
            # Decide whether to continue with fewer models or fail the ensemble
            # break # Option: Stop if one fails

    if not training_successful or len(mlp_instances) < 2: # Need at least 2 for an ensemble
        print("ERROR: Could not train enough MLP instances for the ensemble.")
        return None, None

    print("Individual MLPs for ensemble trained successfully.")

    # Create named estimators list
    estimators_mlp = [(f'mlp_{i+1}', mlp) for i, mlp in enumerate(mlp_instances)]

    # --- Soft Voting MLP Ensemble ---
    mlp_ensemble_soft: Optional[VotingClassifier] = None
    print("Fitting 3x MLP Ensemble (Soft Voting)...")
    start_time = time.time()
    try:
        # n_jobs=1 is crucial for PyTorch within VotingClassifier
        soft_voter = VotingClassifier(estimators=estimators_mlp, voting='soft', n_jobs=1)
        soft_voter.fit(X_train, y_train) # Fit the voter structure
        mlp_ensemble_soft = soft_voter
        print(f"-> 3x MLP Ensemble (Soft) fitted in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"ERROR fitting soft voting MLP ensemble: {e}")

    # --- Hard Voting MLP Ensemble ---
    mlp_ensemble_hard: Optional[VotingClassifier] = None
    print("Fitting 3x MLP Ensemble (Hard Voting)...")
    start_time = time.time()
    try:
        hard_voter = VotingClassifier(estimators=estimators_mlp, voting='hard', n_jobs=1)
        hard_voter.fit(X_train, y_train)
        mlp_ensemble_hard = hard_voter
        print(f"-> 3x MLP Ensemble (Hard) fitted in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"ERROR fitting hard voting MLP ensemble: {e}")

    print("3x MLP Ensemble fitting process complete.")
    return mlp_ensemble_soft, mlp_ensemble_hard


# --- Main Ensemble Training Function (MODIFIED) ---
def train_main_ensembles(
    individual_models: Dict[str, BaseEstimator],
    X_train: np.ndarray,
    y_train: Union[np.ndarray, pd.Series]
    ) -> Tuple[Optional[VotingClassifier], Optional[VotingClassifier]]: # Return Optional
    """
    Creates and fits main soft and hard voting ensembles using selected base models.

    MODIFIED: Includes LR, RF, MLP, and optionally XGBoost and SVM if present in
    `individual_models`. Excludes Decision Tree.

    Args:
        individual_models: A dictionary of pre-trained base models. Keys are model names.
        X_train: The training feature matrix (should be scaled or processed).
        y_train: The training target labels.

    Returns:
        A tuple containing:
         - The fitted soft voting ensemble (VotingClassifier or None if failed).
         - The fitted hard voting ensemble (VotingClassifier or None if failed).
    """
    print("\n--- Training Main Voting Ensembles (LR, RF, MLP, XGB, SVM - Excl. DT) ---")

    # Select base models for the main ensemble
    main_ensemble_keys = ["Logistic Regression", "Random Forest", "PyTorch MLP", "XGBoost", "SVM"]
    estimators_main = []
    for key in main_ensemble_keys:
        if key in individual_models and individual_models[key] is not None:
            # Ensure model seems fitted before adding
            try:
                check_is_fitted(individual_models[key])
                estimators_main.append((key.lower().replace(" ", "_"), individual_models[key]))
                print(f"-> Including '{key}' in main ensemble.")
            except Exception: # Catch if not fitted
                 print(f"WARNING: Model '{key}' not fitted, excluding from main ensemble.")
        else:
             print(f"-> Model '{key}' not available or failed training, excluding from main ensemble.")

    if len(estimators_main) < 2:
        print("ERROR: Not enough valid base models (<2) for the main voting ensemble.")
        return None, None

    # --- Soft Voting Ensemble ---
    ensemble_soft: Optional[VotingClassifier] = None
    print("Fitting Main Ensemble (Soft Voting)...")
    start_time = time.time()
    try:
        # Check if all selected models support predict_proba
        can_do_soft_voting = all(hasattr(model, 'predict_proba') for _, model in estimators_main)
        if not can_do_soft_voting:
            print("WARNING: Not all base models support predict_proba. Skipping soft voting.")
        else:
            # n_jobs=1 recommended if PyTorch wrapper is included
            soft_voter = VotingClassifier(estimators=estimators_main, voting='soft', n_jobs=1)
            soft_voter.fit(X_train, y_train) # Fit the voter structure
            ensemble_soft = soft_voter
            print(f"-> Soft Voting (Main) fitted in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"ERROR fitting main soft voting ensemble: {e}")

    # --- Hard Voting Ensemble ---
    ensemble_hard: Optional[VotingClassifier] = None
    print("Fitting Main Ensemble (Hard Voting)...")
    start_time = time.time()
    try:
        hard_voter = VotingClassifier(estimators=estimators_main, voting='hard', n_jobs=1)
        hard_voter.fit(X_train, y_train)
        ensemble_hard = hard_voter
        print(f"-> Hard Voting (Main) fitted in {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"ERROR fitting main hard voting ensemble: {e}")

    print("Main ensemble fitting process complete.")
    return ensemble_soft, ensemble_hard


# --- Evaluation Function ---
def evaluate_model_performance(
    models_to_evaluate: Dict[str, Optional[BaseEstimator]], # Allow None for failed models
    X_test: np.ndarray,
    y_test: Union[np.ndarray, pd.Series]
    ) -> Dict[str, Optional[Dict[str, float]]]: # Allow None for failed models
    """
    Evaluates the performance of each trained model on the test set.

    Calculates and prints accuracy, precision, recall, F1-score, classification report,
    and confusion matrix for each valid model instance provided.

    Args:
        models_to_evaluate: Dictionary mapping model names to trained model instances
                            (or None if a model failed training).
        X_test: The test feature matrix (should be scaled or PCA-transformed).
        y_test: The test target labels.

    Returns:
        A dictionary storing the evaluation metrics (accuracy, precision, recall, f1)
        for each model name that was successfully evaluated, or None for models
        that failed or were not provided.
    """
    print("\n--- Evaluating Model Performance on Test Set ---")
    results: Dict[str, Optional[Dict[str, float]]] = {} # Store metrics

    for name, model in models_to_evaluate.items():
        # Skip evaluation if model is None (indicating training failure)
        if model is None:
            print(f"\n--- Skipping Evaluation for {name} (Model not available) ---")
            results[name] = None
            continue

        print(f"\n--- {name} Evaluation ---")
        start_eval = time.time()
        try:
            # Ensure model is fitted before predicting
            check_is_fitted(model)
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"ERROR: Prediction failed for model '{name}': {e}")
            results[name] = None # Mark evaluation as failed
            continue # Skip metric calculation for this model

        eval_time = time.time() - start_eval

        # Calculate standard classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        # Use pos_label=1 for clarity in binary classification
        precision = precision_score(y_test, y_pred, zero_division=0, pos_label=1)
        recall = recall_score(y_test, y_pred, zero_division=0, pos_label=1)
        f1 = f1_score(y_test, y_pred, zero_division=0, pos_label=1)

        # Store metrics
        results[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

        # Print detailed results
        print(f"Prediction Time: {eval_time:.4f}s")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f} (Sarcastic=1)")
        print(f"Recall:    {recall:.4f} (Sarcastic=1)")
        print(f"F1-Score:  {f1:.4f} (Sarcastic=1)")
        print("\nClassification Report:")
        try:
            # Ensure target names match expected labels 0 and 1
            print(classification_report(y_test, y_pred, target_names=['Not Sarcastic (0)', 'Sarcastic (1)'], zero_division=0))
        except Exception as report_error:
            print(f"Could not generate classification report: {report_error}")
        print("Confusion Matrix:")
        try:
            cm = confusion_matrix(y_test, y_pred)
            # Handle potential edge case where CM is not 2x2
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            elif cm.size == 1 and len(np.unique(y_test)) == 1: # Only one class predicted and present
                if np.unique(y_test)[0] == 0: tn, fp, fn, tp = cm.item(), 0, 0, 0
                else: tn, fp, fn, tp = 0, 0, 0, cm.item()
            else: # Default or unexpected shape
                 tn, fp, fn, tp = ('N/A',)*4
            print(f"[[TN={tn} FP={fp}]\n [FN={fn} TP={tp}]]")
        except Exception as cm_error:
            print(f"Could not generate confusion matrix: {cm_error}")

    return results


# --- Feature Importance Display Function ---
def display_feature_importance(
    trained_models: Dict[str, Optional[BaseEstimator]], # Allow None
    feature_names: List[str],
    num_top_features: int = 15,
    pca_applied: bool = False
    ) -> None:
    """
    Displays the top feature importances or coefficients for interpretable models.

    Currently supports Decision Tree, Random Forest, Logistic Regression, XGBoost,
    and Linear SVM. Notes limitations for non-linear SVM, MLP, and ensembles.
    Warns if PCA was applied, as results refer to principal components.

    Args:
        trained_models: Dictionary mapping model names to trained model instances (or None).
        feature_names: List of original feature names (before PCA, if applied).
        num_top_features: The number of top features/coefficients to display.
        pca_applied: Flag indicating if PCA was performed before training these models.
    """
    print(f"\n--- Top {num_top_features} Feature Importances/Coefficients ---")

    if pca_applied:
        print("\nWARNING: PCA was applied. Feature metrics below refer to Principal Components.")

    # --- Helper to get appropriate feature names (original or PC names) ---
    def get_current_names(model: BaseEstimator) -> List[str]:
        """Gets feature names, adjusted for PCA if applicable."""
        # Use n_features_in_ which is set by sklearn during fit
        n_features = getattr(model, 'n_features_in_', 0)
        if pca_applied:
            # Generate PC names like PC1, PC2,...
            return [f"PC{i+1}" for i in range(n_features)]
        else:
            # Return original names, ensuring list length matches model's input features
            if n_features <= len(feature_names):
                return feature_names[:n_features]
            else:
                # Fallback if mismatch (shouldn't happen if pipeline is correct)
                print(f"WARNING: Model expected {n_features} features, but only {len(feature_names)} original names available.")
                return [f"Feature_{i+1}" for i in range(n_features)]

    # --- Display logic for each interpretable model type ---
    for name, model in trained_models.items():
        if model is None: continue # Skip models that failed training

        current_names = get_current_names(model)
        importances = None
        coefs = None

        try:
            if name in ["Decision Tree", "Random Forest"] and hasattr(model, 'feature_importances_'):
                importances = pd.Series(model.feature_importances_, index=current_names)
                print(f"\n{name} Top {num_top_features} (Importance):")
                print(importances.nlargest(num_top_features))

            elif name == "XGBoost" and hasattr(model, 'feature_importances_'):
                importances = pd.Series(model.feature_importances_, index=current_names)
                print(f"\n{name} Top {num_top_features} (Importance):")
                print(importances.nlargest(num_top_features))

            elif name == "Logistic Regression" and hasattr(model, 'coef_'):
                coefs = pd.Series(np.abs(model.coef_.flatten()), index=current_names)
                print(f"\n{name} Top {num_top_features} (Absolute Coefficient):")
                print(coefs.nlargest(num_top_features))

            elif name == "SVM" and isinstance(model, SVC) and model.kernel == 'linear' and hasattr(model, 'coef_'):
                 coefs = pd.Series(np.abs(model.coef_.flatten()), index=current_names)
                 print(f"\n{name} (Linear) Top {num_top_features} (Absolute Coefficient):")
                 print(coefs.nlargest(num_top_features))

        except Exception as e:
             print(f"Could not display importance/coefficients for {name}: {e}")


    # --- Notes for non-directly interpretable models ---
    print("\n--- Notes on Model Interpretability ---")
    if "PyTorch MLP" in trained_models:
        print("- Feature importance for PyTorch MLP requires model-specific techniques (e.g., SHAP, LIME, Captum).")
    if "SVM" in trained_models and isinstance(trained_models["SVM"], SVC) and trained_models["SVM"].kernel != 'linear':
        print("- Feature importance for non-linear SVM (RBF/Poly) requires specific techniques (e.g., permutation importance, SHAP).")
    if "Stacking Ensemble" in trained_models or "Ensemble (Soft Voting)" in trained_models or "Ensemble (Hard Voting)" in trained_models:
        print("- Ensemble model importance is complex; it depends on the base models and how they are combined (voting weights or meta-model).")


# --- 7. Main Execution ---

def main():
    """
    Main function to orchestrate the sarcasm detection pipeline.

    Steps:
    1. Configure script parameters (paths, flags).
    2. Set up NLP resources.
    3. Load data (from CSV or raw extraction).
    4. Split data into training and testing sets.
    5. Scale features using StandardScaler.
    6. Optionally apply PCA.
    7. Train individual models (LR, DT, RF, MLP, XGBoost, SVM).
    8. Train ensemble models (3x MLP, Main Voting, Stacking).
    9. Evaluate all trained models on the test set.
    10. Display evaluation summary.
    11. Display feature importances for interpretable models.
    """
    # --- Script Configuration ---
    print("--- Starting Sarcasm Detection Pipeline ---")
    RAW_DATASET_PATH = '../Datasets/Sarcasm_Headlines_Dataset_v2.csv' # <<< ADJUST PATH HERE IF NEEDED
    FORCE_FEATURE_EXTRACTION = False # Set True to ignore CSVs and re-extract
    USE_PCA = False                  # Set True to apply PCA after scaling
    PCA_N_COMPONENTS: Union[int, float, None] = 0.95 # PCA components/variance (e.g., 50 or 0.95)
    TOP_N_FEATURES_DISPLAY = 20      # Number of features for importance reports
    # Flags to control training specific models
    TRAIN_DECISION_TREE = True # Excluded from main ensemble, control if trained at all
    TRAIN_XGBOOST = True
    TRAIN_SVM = True
    TRAIN_MLP_ENSEMBLE = True
    TRAIN_STACKING = True

    # --- Setup ---
    setup_nlp_resources()

    # --- Data Acquisition & Feature Engineering ---
    # `get_data` handles loading pre-computed CSVs or doing full extraction/saving
    X, y = get_data(RAW_DATASET_PATH, force_extract=FORCE_FEATURE_EXTRACTION)
    original_feature_names = X.columns.tolist()

    # --- Data Splitting ---
    print("\n--- Splitting Data ---")
    # Using 25% test split, stratified by target variable 'y'
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing set shape: X={X_test.shape}, y={y_test.shape}")
    print(f"Class distribution in training set:\n{y_train.value_counts(normalize=True)}")
    print(f"Class distribution in test set:\n{y_test.value_counts(normalize=True)}")

    # --- Feature Scaling ---
    print("\n--- Scaling Features ---")
    scaler = StandardScaler() # StandardScaler is generally robust
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features scaled using StandardScaler.")

    # --- Optional PCA ---
    pca_model = None
    if USE_PCA:
        X_train_processed, X_test_processed, pca_model = apply_pca(
            X_train_scaled, X_test_scaled, n_components=PCA_N_COMPONENTS
        )
        current_n_features = pca_model.n_components_
    else:
        # Use scaled data directly if PCA is not applied
        X_train_processed, X_test_processed = X_train_scaled, X_test_scaled
        current_n_features = X_train_scaled.shape[1]

    print(f"\nNumber of features input to models: {current_n_features}")

    # --- Model Training ---
    # Train individual base models (DT training controlled by flag)
    individual_models = train_individual_models(
        X_train_processed, y_train, current_n_features, include_decision_tree=TRAIN_DECISION_TREE
    )

    # Train XGBoost if enabled and available
    if TRAIN_XGBOOST:
        xgb_model = train_xgboost_model(X_train_processed, y_train)
        if xgb_model:
            individual_models["XGBoost"] = xgb_model

    # Train SVM if enabled
    if TRAIN_SVM:
        svm_model = train_svm_model(X_train_processed, y_train)
        if svm_model:
            individual_models["SVM"] = svm_model

    # Train the 3x MLP ensemble if enabled
    mlp_ensemble_soft, mlp_ensemble_hard = (None, None)
    if TRAIN_MLP_ENSEMBLE:
        mlp_ensemble_soft, mlp_ensemble_hard = train_mlp_ensemble(
            X_train_processed, y_train, current_n_features
        )

    # Train the main voting ensembles (using the MODIFIED selection: LR, RF, MLP, XGB, SVM)
    main_ensemble_soft, main_ensemble_hard = train_main_ensembles(
        individual_models, X_train_processed, y_train # Pass dict with all trained individuals
    )

    # Train Stacking Ensemble if enabled
    stacking_model = None
    if TRAIN_STACKING:
        # Define which models go into the stacking base layer
        stacking_base_keys = [
             "Logistic Regression", "Random Forest", "PyTorch MLP", # Core models
             "XGBoost", "SVM" # Optional additions
        ]
        stacking_base_models = { k: individual_models[k] for k in stacking_base_keys if k in individual_models and individual_models[k] is not None }

        if len(stacking_base_models) >= 2: # Need at least 2 diverse models
             stacking_model = train_stacking_model(
                 stacking_base_models, X_train_processed, y_train
                 # meta_model=xgb.XGBClassifier(random_state=42) # Example: Use XGB as meta-model
             )
        else:
            print("Skipping Stacking: Not enough valid base models available/trained.")

    # --- Model Evaluation ---
    # Consolidate all trained models (and ensembles) for evaluation
    all_models_to_evaluate = {**individual_models} # Start with individuals

    # Add ensembles if they were successfully trained
    if main_ensemble_soft: all_models_to_evaluate["Main Ensemble (Soft)"] = main_ensemble_soft
    if main_ensemble_hard: all_models_to_evaluate["Main Ensemble (Hard)"] = main_ensemble_hard
    if TRAIN_MLP_ENSEMBLE:
        if mlp_ensemble_soft: all_models_to_evaluate["3x MLP Ensemble (Soft)"] = mlp_ensemble_soft
        if mlp_ensemble_hard: all_models_to_evaluate["3x MLP Ensemble (Hard)"] = mlp_ensemble_hard
    if TRAIN_STACKING and stacking_model:
        all_models_to_evaluate["Stacking Ensemble"] = stacking_model

    # Evaluate all collected models
    evaluation_results = evaluate_model_performance(
        all_models_to_evaluate, X_test_processed, y_test
    )

    # --- Results Summary ---
    print("\n--- Overall Evaluation Summary ---")
    # Filter out results for models that might have failed evaluation (None values)
    valid_results = {k: v for k, v in evaluation_results.items() if v is not None}
    if valid_results:
        results_df = pd.DataFrame(valid_results).T
        print(results_df[['accuracy', 'precision', 'recall', 'f1']].round(4).sort_values(by='f1', ascending=False))
    else:
        print("No valid evaluation results to display.")

    # --- Feature Importance ---
    # Display importance for the interpretable individual models
    display_feature_importance(
         individual_models, # Pass the dict of individual models
         original_feature_names,
         TOP_N_FEATURES_DISPLAY,
         pca_applied=USE_PCA
    )

    print("\n--- Pipeline Finished ---")


# --- Entry Point ---
if __name__ == "__main__":
    main()