import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split

data = {
    "lyrics": [
        "I'm a rolling thunder, a pouring rain",
        "I see a little silhouetto of a man",
        "Shake it off, shake it off",
        "Back in black, I hit the sack",
        "Take me down to the Paradise City"
    ],
    "labels": [1, 0, 0, 1, 1]
}

dataset = Dataset.from_dict(data)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["lyrics"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Split the dataset into training and evaluation sets
train_dataset, eval_dataset = train_test_split(tokenized_dataset, test_size=0.2)

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