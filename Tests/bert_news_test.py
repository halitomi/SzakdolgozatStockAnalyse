import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'D:/saved_model_news'  # news model path

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# Load dataset
df = pd.read_csv(r'Multi_Agent\\Tests\stock_news_sentiment_300.csv')
df = df.dropna(subset=['Sentence', 'Sentiment'])

# Binary labels: 0 = Negative, 1 = Positive
df['Sentence'] = df['Sentence'].apply(lambda x: x.encode('utf-8', 'ignore').decode('utf-8'))

# Train-test split (just for evaluation)
#_, test_df = train_test_split(df[['Sentence', 'Sentiment']], test_size=1.0, random_state=42)
test_df = df[['Sentence', 'Sentiment']].copy()
# Inference loop
all_preds = []
all_labels = []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    text = row['Sentence']
    label = row['Sentiment']

    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(torch.softmax(outputs.logits, dim=1), dim=1).item()

    all_preds.append(pred)
    all_labels.append(label)

# Results
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Negative", "Positive"]))
