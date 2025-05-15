import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import numpy as np
import os

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
dataset = pd.read_csv('datasets\Sentiment_Stock_data.csv')
dataset['Sentence'] = dataset['Sentence'].fillna('')

# Ensure UTF-8 compatibility for all text
dataset['Sentence'] = dataset['Sentence'].apply(lambda x: x.encode('utf-8', 'ignore').decode('utf-8'))

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    dataset['Sentence'].tolist(), dataset['Sentiment'].tolist(), test_size=0.2, random_state=42
)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a custom Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create train and validation datasets
train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

# DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=16, sampler=RandomSampler(train_dataset))
val_loader = DataLoader(val_dataset, batch_size=16, sampler=SequentialSampler(val_dataset))

# Load the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# Define training arguments with TensorBoard logging setup
logging_dir = './logs/permanent_logs'
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir=logging_dir,
    logging_steps=100,  # Reduced frequency of logging to avoid I/O overload
    save_total_limit=1,  # Keep only the most recent checkpoint
    save_strategy='no',  # Disable saving checkpoints during training
    fp16=True,
)

# Define compute metrics function to include F1 score
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    accuracy = accuracy_score(labels, preds)
    return {'f1': f1, 'accuracy': accuracy}

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Save the trained model and tokenizer
model_save_path = 'D:/saved_model_news'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model saved to {model_save_path}")

# Instructions to run TensorBoard manually
print("To visualize the training progress, run the following command in your terminal:")
print(f"tensorboard --logdir={logging_dir}")
