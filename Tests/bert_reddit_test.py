import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# --- Betöltés ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r'Multi_Agent\\fine_tuned_reddit_model'  # vagy relatív útvonal
model = BertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# --- Adat betöltés ---
df = pd.read_csv("Multi_Agent\Tests\stock__reddit_sentiment_dataset_combined.csv")
df = df.dropna(subset=["Text", "Sentiment"])
label_map = {-1: 0, 0: 1, 1: 2}
df['label'] = df['Sentiment'].map(label_map)

# --- Train-test split ---
#_, test_df = train_test_split(df[['Text', 'label']], test_size=0.3, random_state=42)
test_df = df[['Text', 'label']].copy()

# --- Tesztelés ---
all_preds = []
all_labels = []

for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
    text = row['Text']
    label = row['label']
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    all_preds.append(pred)
    all_labels.append(label)

# --- Eredmények ---
print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Negative", "Neutral", "Positive"]))
