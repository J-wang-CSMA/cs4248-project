# import nltk
# nltk.download('stopwords')

# import nltk
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


from nrclex import NRCLex
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import Normalizer
from torch.utils.data import TensorDataset, DataLoader
import gensim
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC

# Define the eight primary emotions from NRCLex
emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

gensim_model = gensim.models.KeyedVectors.load_word2vec_format('../Datasets/GoogleNews-vectors-negative300.bin', binary=True)



def decontractions(phrase):
    """decontracted takes text and convert contractions into natural form.
     ref: https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490"""
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"won\’t", "will not", phrase)
    phrase = re.sub(r"can\’t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"n\’t", " not", phrase)
    phrase = re.sub(r"\’re", " are", phrase)
    phrase = re.sub(r"\’s", " is", phrase)
    phrase = re.sub(r"\’d", " would", phrase)
    phrase = re.sub(r"\’ll", " will", phrase)
    phrase = re.sub(r"\’t", " not", phrase)
    phrase = re.sub(r"\’ve", " have", phrase)
    phrase = re.sub(r"\’m", " am", phrase)

    return phrase


stopwords = stopwords.words('english')
def preprocess(text_col,stopword):
    preprocessed = []
    for sentence in tqdm(text_col.values):
        # Replace "carriage return" with "space".
        sentence=str(sentence)
        sent = sentence.replace('\\r', ' ')
        #decontraction
        sent=decontractions(sent)
        # Replace "quotes" with "space".
        sent = sent.replace('\\"', ' ')
        # Replace "line feed" with "space".
        sent = sent.replace('\\n', ' ')
        # Replace characters between words with "space".
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        #remove stop words

        if stopword:
            sent = ' '.join(e for e in sent.split() if e not in stopwords)
        else:
           sent = ' '.join(e for e in sent.split())
        # to lowercase
        preprocessed.append(sent.lower().strip())

    return preprocessed

def generate_sentiment_scores(data):
    sid = SentimentIntensityAnalyzer()
    neg=[]
    pos=[]
    neu=[]
    comp=[]
    
    for sentence in tqdm(data): 
        sentence_sentiment_score = sid.polarity_scores(sentence)
        comp.append(sentence_sentiment_score[COMPOUND])
        neg.append(sentence_sentiment_score[NEGATIVE])
        pos.append(sentence_sentiment_score[POSITIVE])
        neu.append(sentence_sentiment_score[NEUTRAL])
    return comp,neg,pos,neu

def generate_emotion_scores(data):

    
    
    # Initialize a dictionary to hold lists for each emotion
    emotion_scores = {emotion: [] for emotion in emotions}
    
    # Iterate over each text input
    for sentence in tqdm(data, desc="Processing"):
        # Create an NRCLex object for the sentence
        emotion = NRCLex(sentence)
        
        # Retrieve the raw emotion scores
        raw_scores = emotion.raw_emotion_scores
        
        # Append the score for each emotion to the corresponding list
        for emotion_name in emotions:
            score = raw_scores.get(emotion_name, 0)
            emotion_scores[emotion_name].append(score)
    
    # Return the scores as a tuple in the order of the defined emotions
    return tuple(emotion_scores[emotion] for emotion in emotions)

def nltk_tokenize_gensim_vectorize(headlines, sentence_len):
    vectorized = []
    for headline in headlines:
        tokens = word_tokenize(headline)
        vectors = []

        for word in tokens:
            if word in gensim_model:
                vectors.append(gensim_model[word])
            else:
                continue  # skip OOV

        # Truncate if too long
        vectors = vectors[:sentence_len]

        # Pad with zero vectors if too short
        while len(vectors) < sentence_len:
            vectors.append(np.zeros(gensim_model.vector_size, dtype=np.float32))
        vectorized.append(np.array(vectors, dtype=np.float32))
    return torch.tensor(np.array(vectorized), dtype=torch.float32)
    


def pad_text_torch(texts, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            # Pad with 0's if sequence is shorter than max_len
            padded_seq = seq + [0] * (max_len - len(seq))
        else:
            # Truncate sequence if it's longer than max_len
            padded_seq = seq[:max_len]
        padded_sequences.append(padded_seq)
    return torch.tensor(padded_sequences, dtype=torch.long)



def normalize_features(X_train, X_test, feature_names):
    """Normalizes multiple features independently using separate Normalizer instances."""
    X_train_norm = []
    X_test_norm = []

    for f in feature_names:
        trans = Normalizer()  # Create a new Normalizer for each feature
        X_train_norm.append(trans.fit_transform(X_train[f].values.reshape(-1, 1)))
        X_test_norm.append(trans.transform(X_test[f].values.reshape(-1, 1)))

    return np.concatenate(X_train_norm, axis=1), np.concatenate(X_test_norm, axis=1)



class DeepCNN1D(nn.Module):
    def __init__(self, dropout = 0.4):
        super(DeepCNN1D, self).__init__()
        

        
        # Convolutional and pooling layers for text
        # in_channels for conv1 must equal the embedding dimension
        self.conv1 = nn.Conv1d(in_channels=300, out_channels=128, kernel_size=8, stride=1, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        # Fully connected layer for numerical input 
        self.fc_num = nn.Linear(len(features), 128)
        
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(64 + 128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc_out = nn.Linear(16, 2)  # 2 classes output
        
    def forward(self, input_comment, input_numerical):
        # Process text input through the embedding layer
        # x1 = self.embedding(input_comment)  # (batch_size, 50, 100)
        x1 = input_comment
        x1 = x1.permute(0, 2, 1)  # (batch_size, 100, 50)

        x1 = F.relu(self.conv1(x1))  # (batch_size, 128, 43)
        x1 = self.pool1(x1)          # (batch_size, 128, 10)
        x1 = F.relu(self.conv2(x1))  # (batch_size, 64, 7)
        x1 = self.pool2(x1)          # (batch_size, 64, 1)
        x1 = torch.flatten(x1, start_dim=1)  # (batch_size, 64)

        # Process numerical input
        x2 = F.relu(self.fc_num(input_numerical))  # (batch_size, 128)

        # Concatenate text and numerical features

        x = torch.cat((x1, x2), dim=1)  # (batch_size, 64+128=192)

        x = self.dropout(x)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_out(x)  # Output logits for 2 classes
        return x


# def main():
CSV_FILE_PATH = '../Datasets/Sarcasm_Headlines_Dataset_v2.csv'
FEATURE = 'headline'
TARGET = 'is_sarcastic'

# List of features to normalize
COMPOUND = "compound"
NEGATIVE = "neg"
POSITIVE = "pos"
NEUTRAL = "neu"
COM_LEN = "com_len"
features = [COMPOUND, NEGATIVE, POSITIVE, NEUTRAL, COM_LEN] + emotions

# Hyperparameters
RANDOM_SEED = 42
MAX_LEN = 50
VOCAB_SIZE = 50
BATCH_SIZE = 512
EPOCHS = 30
DROPOUT = 0.5


# Load the data from CSV
data = pd.read_csv(CSV_FILE_PATH)


data[FEATURE] = preprocess(data[FEATURE], stopword=False)
data[COMPOUND], data[NEGATIVE], data[POSITIVE], data[NEUTRAL] = generate_sentiment_scores(data[FEATURE])
emotion_scores = generate_emotion_scores(data[FEATURE])
for emotion, scores in zip(emotions, emotion_scores):
    data[emotion] = scores
    
data[COM_LEN]=data[FEATURE].apply(lambda x:len(x.split()))
labels = torch.tensor(data[TARGET].values, dtype=torch.long)
one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2)
X=data.drop(TARGET, axis = 1)




# Train-validation split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    X, one_hot_labels, test_size=0.2, random_state=RANDOM_SEED
)

# Add padding
# padded_train, padded_test, tokenizer = text_padding_torch(train_texts[FEATURE], test_texts[FEATURE], max_len=MAX_LEN, vocab_size=VOCAB_SIZE)
# Tokenize

padded_train = nltk_tokenize_gensim_vectorize(train_texts[FEATURE], MAX_LEN)
padded_test = nltk_tokenize_gensim_vectorize(test_texts[FEATURE], MAX_LEN)
numerical_train, numerical_test = normalize_features(train_texts, test_texts, features)

# convert labels to tensor
train_labels_tensor = torch.tensor(np.array(train_labels), dtype=torch.float)
test_labels_tensor = torch.tensor(np.array(test_labels), dtype=torch.float)
numerical_train_tensor = torch.tensor(numerical_train, dtype=torch.float)
numerical_test_tensor = torch.tensor(numerical_test, dtype=torch.float)


# create train loader
train_dataset = TensorDataset(padded_train, numerical_train_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Convert test numerical features to tensor



results = {}
learning_rates = [0.0001, 0.001, 0.01]
epochs = [5, 10, 15]

# winner 0.0001, 15
    
for learning_rate in learning_rates:
    for num_of_epochs in epochs:
        model = DeepCNN1D(dropout = DROPOUT)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        model.train()  # Set the model to training mode
        
        for epoch in range(num_of_epochs):
    
            for batch_text, batch_numerical, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_text, batch_numerical)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
            model.eval()
            
            with torch.no_grad():
                test_outputs = model(padded_test, numerical_test_tensor)
                test_loss = criterion(test_outputs, test_labels_tensor)
                probabilities = F.softmax(test_outputs, dim=1)  # Convert logits to probabilities
                predictions = torch.argmax(probabilities, dim=1)  # Get predicted class indices
            
            print(f"Epoch {epoch+1}/{num_of_epochs} - Training Loss: {loss.item():.4f}, Validation Loss: {test_loss.item():.4f}")
        
    
    
        # Convert to NumPy for metric calculation
        predictions_np = predictions.cpu().numpy()
        true_labels_np = test_labels.argmax(axis=1)  # Convert one-hot labels to class indices
        
        # Compute F1 score
        f1 = f1_score(true_labels_np, predictions_np, average='weighted')  # Weighted for class imbalance
        print(f"F1 Score: {f1:.4f}")
        results[f"{num_of_epochs}, {learning_rate}" ] = f1
        
        with torch.no_grad():
        
            def extract_features(model, text, numerical):
                x1 = model.dropout(text)
                x1 = x1.permute(0, 2, 1)
                x1 = F.relu(model.conv1(x1))
                x1 = model.pool1(x1)
                x1 = F.relu(model.conv2(x1))
                x1 = model.pool2(x1)
                x1 = torch.flatten(x1, start_dim=1)
                x2 = F.relu(model.fc_num(numerical))
                x = torch.cat((x1, x2), dim=1)
                x = F.relu(model.fc1(x))
                x = F.relu(model.fc2(x))
                x = F.relu(model.fc3(x))
                x = F.relu(model.fc4(x))
                return x
            
            train_features = extract_features(model, padded_train, numerical_train_tensor)
            test_features = extract_features(model, padded_test, numerical_test_tensor)
    
            # Step 2: Convert tensors to numpy
        X_train_svm = train_features.cpu().numpy()
        y_train_svm = train_labels_tensor.argmax(dim=1).cpu().numpy()
        
        X_test_svm = test_features.cpu().numpy()
        y_test_svm = test_labels_tensor.argmax(dim=1).cpu().numpy()
        
        # Step 3: Train SVM
        svm = SVC(kernel='linear', C=1.0)
        svm.fit(X_train_svm, y_train_svm)
        
        # Step 4: Predict and evaluate
        y_pred = svm.predict(X_test_svm)
        f1 = f1_score(true_labels_np, y_pred, average='weighted')  # Weighted for class imbalance
    
        print(f"F1 Score: {f1:.4f}")

print(results)

# if __name__ == '__main__':
#     main()