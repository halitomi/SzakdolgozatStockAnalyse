import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, classification_report, recall_score
import numpy as np
import pandas as pd
import torch
import os

# Step 1: Load and Prepare Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = pd.read_csv('datasets\combined_data.csv')  # Adjust the path accordingly

# Map sentiment labels to numerical values
label_map = {-1: 0, 0: 1, 1: 2}  # 0: Negative, 1: Neutral, 2: Positive
data['label'] = data['Sentiment'].map(label_map)

# Convert the pandas dataframe to a Hugging Face Dataset object
dataset = Dataset.from_pandas(data[['Text', 'label']])

# Split the dataset into train and test sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Step 2: Tokenize Dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'f1': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
    }

def tokenize_function(examples):
    return tokenizer(examples['Text'], padding='max_length', truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Step 3: Load Model and Set Training Arguments
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

training_args = TrainingArguments(
    output_dir='./reddit_model_results',
    evaluation_strategy='epoch',  # Keep evaluation at the end of each epoch
    save_strategy='epoch',        # Save model at the end of each epoch to match evaluation strategy
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01, # Load the best model at the end of training
    save_total_limit=2,           # Limit the number of saved models
    logging_dir='./logs',         # Directory for TensorBoard logs
    fp16=True,  
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Step 4: Train the Model
trainer.train()

# Step 5: Save the Model and Tokenizer
model_save_path = r'D:\Models\fine_tuned_reddit_model'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("Model training complete and saved successfully.")
