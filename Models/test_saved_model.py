# evaluate.py
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer, AutoTokenizer
import spacy
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from senticnet.senticnet import SenticNet
import pandas as pd
import warnings
import os
import sys

warnings.filterwarnings("ignore")

SPACY_MODEL = "en_core_web_sm"
CHECKPOINT_DIR = "model_checkpoints"
EVALUATION_PREDICTIONS_FILE = "evaluation_predictions.csv"


print("Initializing spaCy and SenticNet...")
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    print(f"Downloading spaCy model: {SPACY_MODEL}...")
    try:
        spacy.cli.download(SPACY_MODEL)
        nlp = spacy.load(SPACY_MODEL)
    except Exception as e:
        print(f"FATAL: Error downloading/loading spaCy model: {e}")
        sys.exit(1) # Exit if spaCy model cannot be loaded
try:
    sn = SenticNet()
except Exception as e:
    print(f"FATAL: Error initializing SenticNet: {e}")
    sys.exit(1) # Exit if SenticNet cannot be initialized
print("spaCy and SenticNet initialized.")


def get_sentic_score(word):
    """ Retrieves affective polarity from SenticNet, defaults to 0.0 on error. """
    try:
        return float(sn.polarity_value(word))
    except Exception:
        return 0.0

def build_dependency_graph(sentence):
    """ Builds dependency graph using spaCy. """
    try:
        doc = nlp(sentence)
    except Exception as e:
        # print(f"spaCy Error processing sentence: '{sentence[:50]}...'. Error: {e}. Skipping.")
        return [], torch.zeros((0, 0), dtype=torch.float32)
    tokens = [token.text for token in doc]
    n = len(tokens)
    if n == 0: return [], torch.zeros((0, 0), dtype=torch.float32)
    adj = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(adj, 1)
    for token in doc:
        i = token.i
        if token.head.i != i:
            j = token.head.i
            if 0 <= i < n and 0 <= j < n:
                adj[i][j] = 1
                adj[j][i] = 1
    return tokens, torch.tensor(adj)

def build_affective_graph(tokens):
    """ Builds affective graph using SenticNet scores. """
    n = len(tokens)
    if n == 0: return torch.zeros((0, 0), dtype=torch.float32)
    scores = np.array([get_sentic_score(word) for word in tokens], dtype=np.float32)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            adj[i][j] = abs(scores[i] - scores[j])
    return torch.tensor(adj)

# -----------------------------
# GCN Layer Definition
# -----------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear_a = nn.Linear(in_features, out_features)
        self.linear_d = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()
    def forward(self, x, adj):
        adj = adj.to(x.device)
        transformed_features_a = self.linear_a(x)
        convolved_features = torch.matmul(adj, transformed_features_a)
        activated_features = self.relu(convolved_features)
        transformed_features_d = self.linear_d(activated_features)
        output = self.relu(transformed_features_d)
        return output

# -----------------------------
# BertGCNModel Definition
# -----------------------------
class BertGCNModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased',
                 gcn_hidden_dim=300, gcn_layers=4, num_classes=2):
        super(BertGCNModel, self).__init__()
        self.bert_model_name = bert_model_name
        self.gcn_hidden_dim = gcn_hidden_dim
        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained(bert_model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=True)
            if not self.tokenizer.is_fast: print("Warning: Loaded tokenizer is NOT fast.")
        except Exception as e:
            print(f"FATAL: Error loading tokenizer for {bert_model_name}: {e}")
            sys.exit(1)
        bert_hidden_size = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_hidden_size, gcn_hidden_dim)
        self.gcn_layers = nn.ModuleList([
            GCNLayer(gcn_hidden_dim, gcn_hidden_dim) for _ in range(gcn_layers)
        ])
        self.classifier = nn.Linear(gcn_hidden_dim, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def robust_align_bert_spacy(self, sentence, bert_outputs, spacy_tokens):
        bert_embeddings = bert_outputs.last_hidden_state.squeeze(0)
        encoding = self.tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512, padding=True, return_offsets_mapping=True)
        offset_mapping = encoding['offset_mapping'].squeeze(0).cpu().numpy()
        num_spacy_tokens = len(spacy_tokens)
        aligned_embeddings = []
        try:
            doc = nlp(sentence)
            spacy_token_spans = [(token.idx, token.idx + len(token.text)) for token in doc]
            if len(doc) != num_spacy_tokens: spacy_token_spans = None
        except Exception: spacy_token_spans = None
        if spacy_token_spans is None: return None

        bert_token_idx = 0
        for spacy_idx in range(num_spacy_tokens):
            spacy_start, spacy_end = spacy_token_spans[spacy_idx]
            bert_token_indices_for_spacy_token = []
            current_bert_idx_start = bert_token_idx
            while bert_token_idx < len(offset_mapping):
                bert_start, bert_end = offset_mapping[bert_token_idx]
                if bert_start == bert_end:
                    bert_token_idx += 1
                    if bert_token_idx > current_bert_idx_start and not bert_token_indices_for_spacy_token: current_bert_idx_start = bert_token_idx
                    continue
                has_overlap = max(spacy_start, bert_start) < min(spacy_end, bert_end)
                if has_overlap:
                    if bert_token_idx < bert_embeddings.shape[0]: bert_token_indices_for_spacy_token.append(bert_token_idx)
                    if bert_end >= spacy_end: bert_token_idx += 1; break
                    else: bert_token_idx += 1
                elif bert_start >= spacy_end: break
                elif bert_end <= spacy_start:
                    bert_token_idx += 1
                    if bert_token_idx > current_bert_idx_start and not bert_token_indices_for_spacy_token: current_bert_idx_start = bert_token_idx
                else: bert_token_idx += 1
            if bert_token_indices_for_spacy_token:
                valid_indices = [idx for idx in bert_token_indices_for_spacy_token if idx < bert_embeddings.shape[0]]
                if valid_indices: avg_embedding = torch.mean(bert_embeddings[valid_indices, :], dim=0); aligned_embeddings.append(avg_embedding)
                else: aligned_embeddings.append(torch.zeros(self.bert.config.hidden_size, device=self.device))
            else: aligned_embeddings.append(torch.zeros(self.bert.config.hidden_size, device=self.device))
        if not aligned_embeddings: return None
        aligned_tensor = torch.stack(aligned_embeddings)
        if aligned_tensor.shape[0] != num_spacy_tokens: # Adjust shape if needed
             if aligned_tensor.shape[0] > num_spacy_tokens: aligned_tensor = aligned_tensor[:num_spacy_tokens, :]
             else:
                 padding = torch.zeros((num_spacy_tokens - aligned_tensor.shape[0], aligned_tensor.shape[1]), device=self.device)
                 aligned_tensor = torch.cat([aligned_tensor, padding], dim=0)
             if aligned_tensor.shape[0] != num_spacy_tokens: return None
        return aligned_tensor

    def forward(self, sentence):
        if not isinstance(sentence, str) or not sentence.strip(): return torch.zeros(self.num_classes, device=self.device)
        spacy_tokens, dep_adj = build_dependency_graph(sentence)
        if not spacy_tokens: return torch.zeros(self.num_classes, device=self.device)
        aff_adj = build_affective_graph(spacy_tokens)
        num_spacy_tokens = len(spacy_tokens)
        encoding = self.tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512, padding=True)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        # No gradient needed from BERT during evaluation
        with torch.no_grad(): bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_aligned_embeddings = self.robust_align_bert_spacy(sentence, bert_outputs, spacy_tokens)
        if bert_aligned_embeddings is None: return torch.zeros(self.num_classes, device=self.device)
        if dep_adj.size(0) != num_spacy_tokens:
            dep_adj = dep_adj[:num_spacy_tokens, :num_spacy_tokens]
            aff_adj = aff_adj[:num_spacy_tokens, :num_spacy_tokens]
        x = self.fc(bert_aligned_embeddings)
        current_features = x
        for i, layer in enumerate(self.gcn_layers):
            adj_matrix = dep_adj if i % 2 == 0 else aff_adj
            current_features = layer(current_features, adj_matrix)
        final_features = torch.mean(current_features, dim=0)
        logits = self.classifier(final_features)
        return logits

# -----------------------------
# Data Loading Function
# -----------------------------
def load_data_from_csv(filepath, headline_col='headline', label_col='is_sarcastic'):
    """ Loads headlines and labels from a CSV file. """
    if not os.path.exists(filepath): print(f"Error: File not found at {filepath}"); return None, None
    try:
        df = pd.read_csv(filepath)
        if headline_col not in df.columns or label_col not in df.columns:
            print(f"Error: CSV columns '{headline_col}' or '{label_col}' not found."); return None, None
        df = df.dropna(subset=[headline_col, label_col])
        df[headline_col] = df[headline_col].astype(str)
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')
        df = df.dropna(subset=[label_col])
        df[label_col] = df[label_col].astype(int)
        print(f"Loaded {len(df)} records from {filepath}")
        return df[headline_col].tolist(), df[label_col].tolist()
    except Exception as e: print(f"Error reading CSV {filepath}: {e}"); return None, None

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model, sentences, labels, batch_size=32):
    """ Evaluates the model on a given dataset. """
    model.eval()
    all_preds = []
    all_labels_processed = []
    processed_headlines = []
    dataset = list(zip(sentences, labels))
    processed_count = 0

    print(f"Starting evaluation on {len(sentences)} samples...")
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i+batch_size]
            batch_sentences = [s for s, l in batch]
            batch_labels = [l for s, l in batch]
            batch_num = i // batch_size

            current_batch_preds = []
            current_batch_valid_labels = []
            current_batch_valid_headlines = []

            for sentence, label in zip(batch_sentences, batch_labels):
                if not isinstance(sentence, str) or not sentence.strip(): continue
                try:
                    logits = model(sentence)
                    if logits.numel() == 0 or torch.all(logits == 0): continue

                    probabilities = torch.softmax(logits, dim=-1)
                    predicted_class = torch.argmax(probabilities).item()
                    current_batch_preds.append(predicted_class)
                    current_batch_valid_labels.append(label)
                    current_batch_valid_headlines.append(sentence)
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing sentence during evaluation: '{sentence[:50]}...'. Error: {e}")
                    # Decide whether to skip or handle error

            all_preds.extend(current_batch_preds)
            all_labels_processed.extend(current_batch_valid_labels)
            processed_headlines.extend(current_batch_valid_headlines)

            if batch_num % 50 == 0: # Print progress
                 print(f"  Evaluated batch {batch_num}, Processed: {processed_count}/{len(dataset)}")


    accuracy = accuracy_score(all_labels_processed, all_preds) if processed_count > 0 else 0
    f1 = f1_score(all_labels_processed, all_preds, average='weighted', zero_division=0) if processed_count > 0 else 0

    print(f"Evaluation finished. Successfully processed: {processed_count}/{len(sentences)}")
    # Return predictions/labels and corresponding headlines for saving
    return accuracy, f1, all_preds, all_labels_processed, processed_headlines

# -----------------------------
# Main Execution Block for Evaluation
# -----------------------------
if __name__ == "__main__":

    EVAL_CSV_FILE_PATH = 'path/to/your/evaluation_data.csv'

    # SET PATH TO THE TRAINED MODEL CHECKPOINT
    MODEL_FILENAME = "bert_gcn_best_model.pth"
    MODEL_TO_LOAD_PATH = os.path.join(CHECKPOINT_DIR, MODEL_FILENAME)

    # Model Configuration
    BERT_MODEL_NAME = 'bert-base-uncased'
    GCN_HIDDEN_DIM = 300
    GCN_LAYERS = 4
    NUM_CLASSES = 2
    EVAL_BATCH_SIZE = 64

    # --- Load Data ---
    print(f"\nLoading evaluation data from: {EVAL_CSV_FILE_PATH}")
    eval_headlines, eval_labels = load_data_from_csv(EVAL_CSV_FILE_PATH)
    if eval_headlines is None or not eval_headlines:
        print("Exiting: No data loaded for evaluation.")
        sys.exit(1)

    # --- Initialize and Load Model ---
    print("\nInitializing model structure...")
    eval_model = BertGCNModel(
        bert_model_name=BERT_MODEL_NAME,
        gcn_hidden_dim=GCN_HIDDEN_DIM,
        gcn_layers=GCN_LAYERS,
        num_classes=NUM_CLASSES
    )
    print(f"Model device: {eval_model.device}")

    if os.path.exists(MODEL_TO_LOAD_PATH):
        print(f"Loading saved model state from: {MODEL_TO_LOAD_PATH}")
        try:
            state_dict = torch.load(MODEL_TO_LOAD_PATH, map_location=eval_model.device)
            eval_model.load_state_dict(state_dict)
            print("Model state loaded successfully.")
        except Exception as e:
            print(f"FATAL: Error loading model state: {e}. Exiting.")
            sys.exit(1)
    else:
        print(f"FATAL: Model file not found at {MODEL_TO_LOAD_PATH}. Exiting.")
        sys.exit(1)

    # --- Run Evaluation ---
    eval_accuracy, eval_f1, predictions, true_labels, processed_headlines = evaluate_model(
        eval_model, eval_headlines, eval_labels, batch_size=EVAL_BATCH_SIZE
    )

    # --- Display Results ---
    print("\n--- Evaluation Results ---")
    if true_labels:
        print(f"Accuracy: {eval_accuracy:.4f}")
        print(f"F1 Score (Weighted): {eval_f1:.4f}")
        print("\nClassification Report:")
        # Ensure target names match your label encoding (0: Non-Sarcastic, 1: Sarcastic)
        print(classification_report(true_labels, predictions, target_names=['Non-Sarcastic', 'Sarcastic'], zero_division=0))

        # --- Save Predictions to CSV ---
        print(f"\nSaving predictions to {EVALUATION_PREDICTIONS_FILE}...")
        if len(processed_headlines) != len(true_labels):
             print(f"Warning: Mismatch between processed headlines ({len(processed_headlines)}) and labels ({len(true_labels)}). Saving may be incomplete.")
             results_df = pd.DataFrame({'true_label': true_labels, 'predicted_label': predictions})
        else:
             results_df = pd.DataFrame({
                'headline': processed_headlines,
                'true_label': true_labels,
                'predicted_label': predictions
             })
        try:
            results_df.to_csv(EVALUATION_PREDICTIONS_FILE, index=False)
            print(f"Predictions saved successfully.")
        except Exception as e:
            print(f"Error saving predictions to CSV: {e}")
    else:
        print("No samples were successfully processed during evaluation.")

    print("\nEvaluation script finished.")