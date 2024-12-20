import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import random

csv_path = "dataset.csv"
df = pd.read_csv(csv_path)

def preprocess_csv_dataset(row):
    if isinstance(row["track_name"], str) and len(row["track_name"].strip()) > 0:
        return {
            "lyrics": row["track_name"],
            "labels": 1 if row["popularity"] > 50 else 0,
        }
    return None

processed_data = df.apply(preprocess_csv_dataset, axis=1).dropna().tolist()
processed_dataset = Dataset.from_list(processed_data)

random.seed(42)
processed_dataset = processed_dataset.shuffle(seed=42)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["lyrics"], padding="max_length", truncation=True)

tokenized_dataset = processed_dataset.map(tokenize_function, batched=True, batch_size=100)
split_dataset = tokenized_dataset.train_test_split(test_size=0.3)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
results = trainer.evaluate()
print(f"Evaluation Results: {results}")