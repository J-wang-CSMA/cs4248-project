import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer
import spacy
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from senticnet.senticnet import SenticNet
import pandas as pd
import warnings
import os
import copy
import matplotlib.pyplot as plt

# Suppress specific warnings if necessary (e.g., from transformers)
warnings.filterwarnings("ignore")

# -----------------------------
# Initialize spaCy and SenticNet
# -----------------------------
# Check if spaCy model is installed, download if necessary
SPACY_MODEL = "en_core_web_sm"
CHECKPOINT_DIR = "model_checkpoints"
PLOTS_DIR = "plots"
PREDICTIONS_FILE = "test_predictions.csv"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    print(f"Downloading spaCy model: {SPACY_MODEL}...")
    try:
        spacy.cli.download(SPACY_MODEL)
        nlp = spacy.load(SPACY_MODEL)
    except Exception as e:
        print(f"Error downloading/loading spaCy model: {e}")
        print("Please ensure you have internet access and sufficient permissions.")
        exit()

try:
    sn = SenticNet()
except Exception as e:
    print(f"Error initializing SenticNet: {e}")
    print("Please ensure the 'senticnet' library is correctly installed ('pip install senticnet').")
    exit()


def get_sentic_score(word):
    """
    Retrieve the affective polarity value for a word using SenticNet.
    If the word is not found or an error occurs, returns 0.0.
    """
    try:
        score = sn.polarity_value(word)
        return float(score)
    except Exception:
        return 0.0


# -----------------------------
# Graph Construction Functions
# -----------------------------
def build_dependency_graph(sentence):
    """
    Use spaCy to obtain the dependency parse and create an adjacency matrix.
    Returns:
      - tokens: list of token texts (in order)
      - dep_adj: torch.FloatTensor of shape (n, n)
    """
    try:
        doc = nlp(sentence)
    except Exception as e:
        print(f"spaCy Error processing sentence: '{sentence}'. Error: {e}. Skipping.")
        return [], torch.zeros((0, 0), dtype=torch.float32)

    tokens = [token.text for token in doc]
    n = len(tokens)
    if n == 0:
        return [], torch.zeros((0, 0), dtype=torch.float32)

    adj = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(adj, 1)  # Add self-loops for dependency graph
    for token in doc:
        i = token.i
        # Connect token with its head (if not itself)
        if token.head.i != i:
            j = token.head.i
            if 0 <= i < n and 0 <= j < n:
                adj[i][j] = 1
                adj[j][i] = 1  # Ensure the graph is undirected.

    return tokens, torch.tensor(adj)


def build_affective_graph(tokens):
    """
    Create an affective graph where each edge weight is the absolute difference
    of affective scores (from SenticNet) for the two words.
    Self-loops according to the formula abs(score[i]-score[i]) are 0.
    Returns:
      - aff_adj: torch.FloatTensor of shape (n, n)
    """
    n = len(tokens)
    if n == 0:
        return torch.zeros((0, 0), dtype=torch.float32)
    scores = np.array([get_sentic_score(word) for word in tokens], dtype=np.float32)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            adj[i][j] = abs(scores[i] - scores[j])
    return torch.tensor(adj)


# -----------------------------
# GCN Layer Implementation (Revised based on Eq. 4 interpretation)
# -----------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear_a = nn.Linear(in_features, out_features)
        self.linear_d = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        # Ensure adjacency matrix is on the same device as input features
        adj = adj.to(x.device)

        # Implementation attempting to match Eq. 4 in the paper:
        # g^l = ReLU(ReLU(A @ (g^{l-1} @ W_a + b_a)) @ W_d + b_d)
        # 1. Transform features: g^{l-1} @ W_a + b_a
        transformed_features_a = self.linear_a(x)  # Shape: (n, out_features)
        # 2. Graph convolution A @ (transformed features)
        convolved_features = torch.matmul(adj, transformed_features_a)  # Shape: (n, out_features)
        # 3. First ReLU
        activated_features = self.relu(convolved_features)  # Shape: (n, out_features)
        # 4. Transform again: activated_features @ W_d + b_d
        transformed_features_d = self.linear_d(activated_features)  # Shape: (n, out_features)
        # 5. Final ReLU
        output = self.relu(transformed_features_d)  # Shape: (n, out_features)
        return output


# -----------------------------
# BERT-GCN Model
# -----------------------------
class BertGCNModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased',
                 gcn_hidden_dim=300, gcn_layers=4, num_classes=2):
        super(BertGCNModel, self).__init__()
        self.bert_model_name = bert_model_name
        self.gcn_hidden_dim = gcn_hidden_dim
        self.num_classes = num_classes

        self.bert = BertModel.from_pretrained(bert_model_name)

        print(f"Loading FAST tokenizer for: {bert_model_name}")
        try:
            # Load the fast tokenizer explicitly
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=True)
            print(f"Successfully loaded tokenizer: {type(self.tokenizer)}")
            if not self.tokenizer.is_fast:
                print("Warning: Loaded tokenizer is NOT flagged as fast, offset mapping might still fail.")
            else:
                print("Tokenizer is FAST, offset mapping should be supported.")

        except Exception as e:
            print(f"Error loading tokenizer with AutoTokenizer: {e}")
            print("Please ensure 'transformers' and 'tokenizers' libraries are installed and updated.")
            print("Falling back to BertTokenizer (might cause offset mapping error)...")
            # Fallback just in case
            self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, use_fast=True)

        bert_hidden_size = self.bert.config.hidden_size

        self.fc = nn.Linear(bert_hidden_size, gcn_hidden_dim)
        self.gcn_layers = nn.ModuleList([
            GCNLayer(gcn_hidden_dim, gcn_hidden_dim) for _ in range(gcn_layers)
        ])
        self.classifier = nn.Linear(gcn_hidden_dim, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def robust_align_bert_spacy(self, sentence, bert_outputs, spacy_tokens):
        """
        Aligns BERT subword embeddings to spaCy word tokens using offset mapping
        and averages embeddings for subwords belonging to the same spaCy token.

        Args:
            sentence (str): The original input sentence.
            bert_outputs: The output from the BERT model (including last_hidden_state).
            spacy_tokens (list): A list of spaCy token text strings.

        Returns:
            torch.Tensor: Aligned embeddings tensor of shape (num_spacy_tokens, bert_hidden_size).
                         Returns None if alignment fails or results in empty tensor.
        """
        # Get BERT embeddings and offset mapping
        bert_embeddings = bert_outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_size)
        encoding = self.tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512, padding=True,
                                  return_offsets_mapping=True)
        offset_mapping = encoding['offset_mapping'].squeeze(0).cpu().numpy()  # (seq_len, 2)

        num_spacy_tokens = len(spacy_tokens)
        aligned_embeddings = []

        # Use spaCy's doc object to get character indices easily
        try:
            doc = nlp(sentence)
            if len(doc) != num_spacy_tokens:
                # Fallback if spaCy tokenization differs unexpectedly from the list passed
                print("Warning: spaCy token count mismatch during alignment. Using provided token list length.")
                spacy_token_spans = None  # Cannot reliably get spans if doc doesn't match
            else:
                spacy_token_spans = [(token.idx, token.idx + len(token.text)) for token in doc]
        except Exception as e:
            print(f"spaCy error during alignment for sentence '{sentence[:50]}...': {e}")
            return None  # Cannot align without spaCy spans

        bert_idx = 0
        for spacy_idx in range(num_spacy_tokens):
            if spacy_token_spans is None:  # If fallback occurred
                # Cannot perform alignment without reliable spaCy spans
                print("Error: Cannot align tokens due to spaCy mismatch.")
                return None

            spacy_start, spacy_end = spacy_token_spans[spacy_idx]

            # Find corresponding BERT tokens using offset mapping
            bert_token_indices_for_spacy_token = []
            while bert_idx < len(offset_mapping):
                bert_start, bert_end = offset_mapping[bert_idx]

                # Skip special tokens [CLS], [SEP], [PAD] which have (0, 0) offset
                if bert_start == 0 and bert_end == 0:
                    bert_idx += 1
                    continue

                # Check for overlap: if BERT token span is within or overlaps spaCy token span
                # Condition: max(spacy_start, bert_start) < min(spacy_end, bert_end)
                if max(spacy_start, bert_start) < min(spacy_end, bert_end):
                    bert_token_indices_for_spacy_token.append(bert_idx)
                    # If the BERT token ends at or after the spaCy token, move to the next spaCy token
                    if bert_end >= spacy_end:
                        bert_idx += 1  # Ensure bert_idx advances
                        break  # Move to next spacy token
                    else:
                        # BERT token is fully contained, continue checking next BERT token for this spaCy token
                        bert_idx += 1
                # If BERT token starts after the current spaCy token ends, move to next spaCy token
                elif bert_start >= spacy_end:
                    break  # Move to next spacy token
                # If BERT token ends before or at the start of the spaCy token, advance BERT index
                elif bert_end <= spacy_start:
                    bert_idx += 1
                else:  # Should not happen with contiguous spans? Advance bert_idx just in case
                    bert_idx += 1

            # Aggregate embeddings for the found BERT tokens
            if bert_token_indices_for_spacy_token:
                # Select embeddings corresponding to these indices
                # Need to ensure indices are valid for bert_embeddings tensor
                valid_indices = [idx for idx in bert_token_indices_for_spacy_token if idx < bert_embeddings.shape[0]]
                if valid_indices:
                    token_embeddings = bert_embeddings[valid_indices, :]
                    # Average the embeddings
                    avg_embedding = torch.mean(token_embeddings, dim=0)
                    aligned_embeddings.append(avg_embedding)
                else:
                    # If indices somehow became invalid, append zeros
                    print(f"Warning: Invalid BERT indices found for spaCy token {spacy_idx}. Appending zero vector.")
                    aligned_embeddings.append(torch.zeros(self.bert.config.hidden_size, device=self.device))
            else:
                # Handle cases where no BERT token maps (e.g., multiple spaces collapsed by spaCy)
                # Append a zero vector or handle appropriately
                aligned_embeddings.append(torch.zeros(self.bert.config.hidden_size, device=self.device))

        if not aligned_embeddings:
            return None

        # Stack the aggregated embeddings
        aligned_tensor = torch.stack(aligned_embeddings)  # Shape: (num_spacy_tokens, bert_hidden_size)

        # Final check for expected shape
        if aligned_tensor.shape[0] != num_spacy_tokens:
            print(
                f"Warning: Final aligned tensor shape mismatch ({aligned_tensor.shape[0]} vs {num_spacy_tokens}). Returning None.")
            return None

        return aligned_tensor

    def forward(self, sentence):
        """ Forward pass for a single sentence string. """
        if not isinstance(sentence, str) or not sentence.strip():
            return torch.zeros(self.num_classes, device=self.device)

        # Graph Construction
        spacy_tokens, dep_adj = build_dependency_graph(sentence)
        if not spacy_tokens:
            return torch.zeros(self.num_classes, device=self.device)
        aff_adj = build_affective_graph(spacy_tokens)
        num_spacy_tokens = len(spacy_tokens)

        # BERT Feature Extraction
        encoding = self.tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512, padding=True)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # --- Token Alignment ---
        bert_aligned_embeddings = self.robust_align_bert_spacy(sentence, bert_outputs, spacy_tokens)

        if bert_aligned_embeddings is None or bert_aligned_embeddings.shape[0] != num_spacy_tokens:
            # Handle alignment failure: return zero logits or manage error
            # print(f"Alignment failed or shape mismatch for sentence: {sentence[:50]}...")
            return torch.zeros(self.num_classes, device=self.device)

        # Ensure graphs match the number of spacy tokens (should be guaranteed if alignment succeeded)
        if dep_adj.size(0) != num_spacy_tokens:
            dep_adj = dep_adj[:num_spacy_tokens, :num_spacy_tokens]
            aff_adj = aff_adj[:num_spacy_tokens, :num_spacy_tokens]

        # BERT embeddings to GCN dimension
        x = self.fc(bert_aligned_embeddings)  # (num_spacy_tokens, gcn_hidden_dim)

        # GCN Propagation
        current_features = x
        for i, layer in enumerate(self.gcn_layers):
            adj_matrix = dep_adj if i % 2 == 0 else aff_adj
            current_features = layer(current_features, adj_matrix)

        # Aggregate node features (Assumption: mean pooling)
        final_features = torch.mean(current_features, dim=0)

        # Classify
        logits = self.classifier(final_features)
        return logits


# -----------------------------
# Data Loading Function
# -----------------------------
def load_data_from_csv(filepath, headline_col='headline', label_col='is_sarcastic'):
    """
    Loads headlines and labels from a CSV file.
    Args:
        filepath (str): Path to the CSV file.
        headline_col (str): Name of the column containing headlines.
        label_col (str): Name of the column containing labels (0 or 1).
    Returns:
        tuple: (list of headlines, list of labels) or (None, None) if error.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None, None

    try:
        df = pd.read_csv(filepath)
        # Basic validation
        if headline_col not in df.columns or label_col not in df.columns:
            print(f"Error: CSV must contain columns named '{headline_col}' and '{label_col}'.")
            print(f"Found columns: {df.columns.tolist()}")
            return None, None

        # Ensure data types are correct and handle potential missing values
        df = df.dropna(subset=[headline_col, label_col])
        df[headline_col] = df[headline_col].astype(str)
        df[label_col] = pd.to_numeric(df[label_col], errors='coerce')  # Convert label to numeric, coerce errors to NaN
        df = df.dropna(subset=[label_col])  # Drop rows where label conversion failed
        df[label_col] = df[label_col].astype(int)  # Convert valid labels to int

        headlines = df[headline_col].tolist()
        labels = df[label_col].tolist()

        print(f"Loaded {len(headlines)} headlines from {filepath}")
        return headlines, labels

    except Exception as e:
        print(f"Error reading CSV file {filepath}: {e}")
        return None, None


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model, sentences, labels, batch_size=32):
    """ Evaluates the model on a given dataset. """
    model.eval() # Set model to evaluation mode
    total_loss = 0
    all_preds = []
    all_labels_processed = []
    processed_count = 0
    criterion = nn.CrossEntropyLoss()
    dataset = list(zip(sentences, labels))

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i : i+batch_size]
            batch_sentences = [s for s, l in batch]
            batch_labels = [l for s, l in batch]

            batch_logits = []
            valid_labels = []

            for sentence, label in zip(batch_sentences, batch_labels):
                 if not isinstance(sentence, str) or not sentence.strip(): continue
                 logits = model(sentence)
                 if logits.numel() == 0 or torch.all(logits == 0): continue
                 batch_logits.append(logits)
                 valid_labels.append(label)

            if not batch_logits: continue

            try:
                batch_logits_tensor = torch.stack(batch_logits)
                batch_labels_tensor = torch.tensor(valid_labels, dtype=torch.long).to(model.device)

                loss = criterion(batch_logits_tensor, batch_labels_tensor)
                total_loss += loss.item()

                preds = torch.argmax(batch_logits_tensor, dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels_processed.extend(valid_labels)
                processed_count += len(valid_labels)
            except RuntimeError as e:
                 print(f"RuntimeError during evaluation batch starting at index {i}: {e}")
                 continue # Skip batch on error


    avg_loss = total_loss / ((processed_count + batch_size -1)// batch_size) if processed_count > 0 else 0
    accuracy = accuracy_score(all_labels_processed, all_preds) if processed_count > 0 else 0
    f1 = f1_score(all_labels_processed, all_preds, average='weighted', zero_division=0) if processed_count > 0 else 0

    return avg_loss, accuracy, f1, all_preds, all_labels_processed # Return predictions/labels too


# -----------------------------
# Training Function
# -----------------------------
def train_model(model, train_sentences, train_labels, test_sentences, test_labels,
                epochs=5, lr=1e-4, batch_size=16,
                early_stopping_patience=5, early_stopping_delta=0.001,
                checkpoint_freq=5):
    """
    Trains the model with evaluation per epoch, early stopping, and checkpointing.
    """
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_dataset = list(zip(train_sentences, train_labels))

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_test_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_model_epoch = -1

    print(f"\nStarting Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train() # Set model to training mode for the epoch
        total_train_loss = 0
        all_train_preds = []
        all_train_labels_in_epoch = []
        processed_train_count = 0
        batch_num = 0

        # --- Training Loop for one epoch ---
        for i in range(0, len(train_dataset), batch_size):
            batch = train_dataset[i : i+batch_size]
            batch_sentences = [s for s, l in batch]
            batch_labels = [l for s, l in batch]
            batch_num += 1
            optimizer.zero_grad()
            batch_logits = []
            valid_labels = []
            for sentence, label in zip(batch_sentences, batch_labels):
                 if not isinstance(sentence, str) or not sentence.strip(): continue
                 logits = model(sentence)
                 if logits.numel() == 0 or torch.all(logits == 0): continue
                 batch_logits.append(logits)
                 valid_labels.append(label)
            if not batch_logits: continue

            try:
                batch_logits_tensor = torch.stack(batch_logits)
                batch_labels_tensor = torch.tensor(valid_labels, dtype=torch.long).to(model.device)
                loss = criterion(batch_logits_tensor, batch_labels_tensor)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                preds = torch.argmax(batch_logits_tensor, dim=-1).cpu().tolist()
                all_train_preds.extend(preds)
                all_train_labels_in_epoch.extend(valid_labels)
                processed_train_count += len(valid_labels)
                if batch_num % 100 == 0:
                    print(f"  Epoch {epoch + 1}/{epochs}, Batch {batch_num}, Processed: {processed_train_count}/{len(train_dataset)}")
            except RuntimeError as e:
                print(f"RuntimeError during training batch {batch_num}: {e}")
                continue # Skip batch

        # --- Calculate Training Metrics for Epoch ---
        avg_train_loss = total_train_loss / batch_num if batch_num > 0 else 0
        train_acc = accuracy_score(all_train_labels_in_epoch, all_train_preds) if processed_train_count > 0 else 0
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)

        # --- Evaluation on Test Set for Epoch ---
        test_loss, test_acc, test_f1, _, _ = evaluate_model(model, test_sentences, test_labels, batch_size)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch + 1}/{epochs} Summary - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

        # --- Checkpoint Saving ---
        if (epoch + 1) % checkpoint_freq == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"bert_gcn_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

        # --- Early Stopping Check ---
        if test_loss < best_test_loss - early_stopping_delta:
            best_test_loss = test_loss
            epochs_no_improve = 0
            # Deep copy the model state dict when a new best is found
            best_model_state = copy.deepcopy(model.state_dict())
            best_model_epoch = epoch + 1
            print(f"  New best test loss: {best_test_loss:.4f}. Saving best model state.")
        else:
            epochs_no_improve += 1
            print(f"  Test loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    # --- End of Training ---
    print("\n--- Training Finished ---")
    # Save the best model explicitly if training finished normally or early stopping occurred
    best_model_path = None
    if best_model_state:
        best_model_path = os.path.join(CHECKPOINT_DIR, "bert_gcn_best_model.pth")
        torch.save(best_model_state, best_model_path)
        print(f"Best model state (from epoch {best_model_epoch} with loss {best_test_loss:.4f}) saved to {best_model_path}")
    else:
        # If training finished without improvement or only 1 epoch, save the last state as 'best'
        print("No improvement detected or training too short; saving last model state as best.")
        best_model_path = os.path.join(CHECKPOINT_DIR, "bert_gcn_last_model.pth")
        torch.save(model.state_dict(), best_model_path)

    return history, best_model_path


# -----------------------------
# Plotting Function
# -----------------------------
def plot_metrics(history):
    """ Generates and saves plots for training/test loss and accuracy. """
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['test_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_filename = os.path.join(PLOTS_DIR, "training_metrics.png")
    plt.savefig(plot_filename)
    print(f"Metrics plot saved to {plot_filename}")


# -----------------------------
# Main Execution Block
# -----------------------------
if __name__ == "__main__":
    # --- Configuration ---
    CSV_FILE_PATH = '../Datasets/Sarcasm_Headlines_Dataset_v2.csv'
    BERT_MODEL_NAME = 'bert-base-uncased'
    GCN_HIDDEN_DIM = 300
    GCN_LAYERS = 4
    NUM_CLASSES = 2
    # Training Params
    EPOCHS = 50 # Max epochs
    LEARNING_RATE = 0.01 # Adjusted LR
    BATCH_SIZE = 64
    TEST_SPLIT_SIZE = 0.2
    # Early Stopping Params
    EARLY_STOPPING_PATIENCE = 5 # Stop if no improvement for 5 epochs
    EARLY_STOPPING_DELTA = 0.001 # Minimum improvement in test loss
    # Checkpointing Params
    CHECKPOINT_FREQ = 5 # Save every 5 epochs

    # --- Data Loading & Splitting ---
    print(f"Loading data from: {CSV_FILE_PATH}")
    headlines, labels = load_data_from_csv(CSV_FILE_PATH)
    if headlines is None or not headlines: exit()
    train_headlines, test_headlines, train_labels, test_labels = train_test_split(
        headlines, labels, test_size=TEST_SPLIT_SIZE, random_state=42, stratify=labels
    )
    print(f"Data split: {len(train_headlines)} train, {len(test_headlines)} test.")

    # --- Model Initialization ---
    print("Initializing model...")
    model = BertGCNModel(
        bert_model_name=BERT_MODEL_NAME, gcn_hidden_dim=GCN_HIDDEN_DIM,
        gcn_layers=GCN_LAYERS, num_classes=NUM_CLASSES
    )
    print(f"Model on device: {model.device}. Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Training ---
    history, best_model_path = train_model(
        model, train_headlines, train_labels, test_headlines, test_labels,
        epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_delta=EARLY_STOPPING_DELTA,
        checkpoint_freq=CHECKPOINT_FREQ
    )

    # --- Plotting ---
    if history['train_loss']:
        plot_metrics(history)
    else:
        print("Skipping plotting as no training history was recorded.")

    # --- Final Evaluation & Saving Predictions ---
    print("\n--- Final Evaluation on Test Set using Best Model ---")
    if best_model_path and os.path.exists(best_model_path):
        print(f"Loading best model state from: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("Warning: Best model path not found or training didn't run. Evaluating with the last model state.")

    # Run evaluation
    final_test_loss, final_test_acc, final_test_f1, test_predictions, processed_test_labels = evaluate_model(
        model, test_headlines, test_labels, batch_size=BATCH_SIZE
    )

    print("\n--- Final Test Set Results (Best Model) ---")
    if processed_test_labels: # Check if evaluation produced results
         print(f"Accuracy: {final_test_acc:.4f}")
         print(f"F1 Score (Weighted): {final_test_f1:.4f}")
         print(f"Loss: {final_test_loss:.4f}")

         # --- Save Predictions to CSV ---
         results_df = pd.DataFrame({
             'headline': [h for h, l in zip(test_headlines, test_labels) if isinstance(h, str) and h.strip()], # Get original headlines corresponding to processed labels
             'true_label': processed_test_labels,
             'predicted_label': test_predictions
         })
         if len(results_df) != len(processed_test_labels):
              print(f"Warning: Mismatch between number of processed labels ({len(processed_test_labels)}) and potential headlines ({len(results_df)}). Predictions CSV might be incomplete.")
              results_df = pd.DataFrame({'true_label': processed_test_labels, 'predicted_label': test_predictions})


         try:
            results_df.to_csv(PREDICTIONS_FILE, index=False)
            print(f"Test predictions saved to {PREDICTIONS_FILE}")
         except Exception as e:
            print(f"Error saving predictions to CSV: {e}")

    else:
         print("No test samples were successfully processed for final evaluation.")


    # --- Example Prediction (using best model) ---
    print("\n--- Example Prediction (Best Model) ---")
    test_sentence = "Reading this dry manual is my absolute favorite way to spend a Saturday afternoon."
    model.eval()
    with torch.no_grad():
        try:
             logits = model(test_sentence)
             if logits.numel() > 0 and not torch.all(logits == 0):
                 probabilities = torch.softmax(logits, dim=-1)
                 predicted_class = torch.argmax(probabilities).item()
                 print(f"Test Sentence: '{test_sentence}'")
                 print(f"Predicted Class: {predicted_class} (0: Non-Sarcastic, 1: Sarcastic)")
                 print(f"Probabilities: {probabilities.cpu().numpy()}")
             else: print(f"Could not get valid prediction for: '{test_sentence}'")
        except Exception as e: print(f"Error during example prediction: {e}")
