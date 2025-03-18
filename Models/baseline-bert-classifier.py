from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

def load_data(filepath, feature, target):
    df = pd.read_csv(filepath)
    feature_col = df[feature]
    target_col = df[target]
    return feature_col, target_col


def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        logit_values = outputs.logits

        _, predictions = torch.max(logit_values, dim=1)
        correct_predictions += torch.sum(predictions == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()

    epoch_acc = correct_predictions.double() / len(data_loader.dataset)
    epoch_loss = np.mean(losses)
    return epoch_acc, epoch_loss


def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    predictions_all = []
    labels_all = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logit_values = outputs.logits
            _, predictions = torch.max(logit_values, dim=1)

            predictions_all.extend(predictions.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
            losses.append(loss.item())

    accuracy = accuracy_score(labels_all, predictions_all)
    f1 = f1_score(labels_all, predictions_all, average='weighted')
    avg_loss = np.mean(losses)
    return accuracy, f1, avg_loss


def main():
    CSV_FILE_PATH = '../Datasets/Sarcasm_Headlines_Dataset_v2.csv'
    FEATURE = 'headline'
    TARGET = 'is_sarcastic'

    # Hyperparameters
    RANDOM_SEED = 42
    MAX_LEN = 64
    BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.03

    torch.manual_seed(RANDOM_SEED)

    # Load the data from CSV
    headlines, labels = load_data(CSV_FILE_PATH, FEATURE, TARGET)

    # Train-validation split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        headlines, labels, test_size=0.2, random_state=RANDOM_SEED
    )
    # Reset the indices so we can safely index them later
    train_texts = train_texts.reset_index(drop=True)
    val_texts = val_texts.reset_index(drop=True)
    train_labels = train_labels.reset_index(drop=True)
    val_labels = val_labels.reset_index(drop=True)

    print("Sample training headline:", train_texts[0])
    print("Corresponding label:", train_labels[0])
    print(f"Number of training samples: {len(train_texts)}")
    print(f"Number of validation samples: {len(val_texts)}")

    # Tokenizer and Model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Tokenize training data
    # Convert to Python lists first
    train_texts_list = train_texts.tolist()
    print("\nTokenizing training data...")
    train_encodings = tokenizer(
        train_texts_list,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_attention_mask=True,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    print("Training input_ids shape:", train_encodings["input_ids"].shape)
    print("Training attention_mask shape:", train_encodings["attention_mask"].shape)

    # Convert labels to a tensor
    train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)
    print("Training labels shape:", train_labels_tensor.shape)

    # Tokenize validation data
    val_texts_list = val_texts.tolist()
    print("\nTokenizing validation data...")
    val_encodings = tokenizer(
        val_texts_list,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_attention_mask=True,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    print("Validation input_ids shape:", val_encodings["input_ids"].shape)
    print("Validation attention_mask shape:", val_encodings["attention_mask"].shape)

    # Convert labels to a tensor
    val_labels_tensor = torch.tensor(val_labels.values, dtype=torch.long)
    print("Validation labels shape:", val_labels_tensor.shape)

    # Create DataLoaders
    train_dataset = TensorDataset(
        train_encodings["input_ids"],
        train_encodings["attention_mask"],
        train_labels_tensor
    )
    val_dataset = TensorDataset(
        val_encodings["input_ids"],
        val_encodings["attention_mask"],
        val_labels_tensor
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Training Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        # Training
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")

        # Evaluation
        val_acc, val_f1, val_loss = eval_model(model, val_loader, device)
        print(f"Val   Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f} | Val F1: {val_f1:.4f}")


    model.save_pretrained('./sarcasm_bert_model/bert-base-sarcasm-tuned')
    tokenizer.save_pretrained('./sarcasm_bert_model/tokenizer_sarcasm_tuned')
    print("\nModel and tokenizer have been saved to './sarcasm_bert_model'.")


if __name__ == '__main__':
    main()
