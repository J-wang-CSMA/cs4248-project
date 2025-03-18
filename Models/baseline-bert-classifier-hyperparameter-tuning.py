import optuna
import torch
from torch.optim import AdamW
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# Data Preparation
CSV_FILE_PATH = '../Datasets/Sarcasm_Headlines_Dataset_v2.csv'
FEATURE = 'headline'
TARGET = 'is_sarcastic'

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

df = pd.read_csv(CSV_FILE_PATH)
headlines = df[FEATURE]
labels = df[TARGET]

train_texts, val_texts, train_labels, val_labels = train_test_split(
    headlines, labels, test_size=0.2, random_state=RANDOM_SEED
)
train_texts = train_texts.reset_index(drop=True)
val_texts = val_texts.reset_index(drop=True)
train_labels = train_labels.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

MAX_LEN = 64

# Tokenizing the input features
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("Tokenizing train data...")
train_encodings = tokenizer(
    train_texts.tolist(),
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
train_labels_tensor = torch.tensor(train_labels.values, dtype=torch.long)

print("Tokenizing validation data...")
val_encodings = tokenizer(
    val_texts.tolist(),
    add_special_tokens=True,
    max_length=MAX_LEN,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
val_labels_tensor = torch.tensor(val_labels.values, dtype=torch.long)

# Create datasets
train_dataset = TensorDataset(
    train_encodings['input_ids'],
    train_encodings['attention_mask'],
    train_labels_tensor
)
val_dataset = TensorDataset(
    val_encodings['input_ids'],
    val_encodings['attention_mask'],
    val_labels_tensor
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Helper functions for training and validation
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0

    for batch in loader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logit_values = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predictions = torch.max(logit_values, dim=1)
        correct += torch.sum(predictions == labels)

    avg_loss = total_loss / len(loader)
    accuracy = correct.double() / len(loader.dataset)
    return avg_loss, accuracy


def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logit_values = outputs.logits

            total_loss += loss.item()
            _, predictions = torch.max(logit_values, dim=1)

            correct += torch.sum(predictions == labels)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = correct.double() / len(loader.dataset)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    return avg_loss, accuracy.item(), f1


# Optuna objective function with hyperparameters to search
def objective(trial):
    # Hyperparameters
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 5e-5])
    epochs = trial.suggest_int("epochs", 2, 5, step=1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_accuracy = 0.0
    best_val_f1 = 0.0

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        # Evaluate
        val_loss, val_acc, val_f1 = evaluate(model, val_loader)

        # Track best
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_val_f1 = val_f1

    trial.set_user_attr("best_val_f1", best_val_f1)

    return best_val_accuracy


# Run optuna study to optimize hyperparameters
def main():
    # Create a study that tries to maximize the returned metric (val accuracy)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("\nOptimization finished. Best trial:")
    best_trial = study.best_trial
    print(f"  Value (best validation accuracy): {best_trial.value:.4f}")
    print(f"  Params:")
    for key, val in best_trial.params.items():
        print(f"    {key}: {val}")
    print(f"  Best validation F1 (stored in user_attr): {best_trial.user_attrs.get('best_val_f1')}")

    # Retrieve all trials in a pandas DataFrame
    df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    print("\nAll trials DataFrame:")
    print(df)


if __name__ == "__main__":
    main()
