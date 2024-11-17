import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

ds = load_dataset("maharshipandya/spotify-tracks-dataset")

def preprocess_spotify_dataset(example):
    return {
        "lyrics": example["track_name"],
        "labels": 1 if example["popularity"] > 50 else 0
    }

processed_ds = ds["train"].map(preprocess_spotify_dataset, remove_columns=["track_name", "popularity"])

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(list(examples["lyrics"]), padding="max_length", truncation=True)

tokenized_dataset = processed_ds.map(tokenize_function, batched=True)

split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
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