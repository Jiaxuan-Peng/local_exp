#/home/u1413911/micromamba/bin/python /home/u1413911/.micromamba/local_exp/DeBERTa-v3-base.py
#pip install datasets transformers
#pip install git-lfs
import transformers
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
import evaluate
import torch
import numpy as np
import argparse
import os
import jsonlines



output_dir = "./deberta_imdb"
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
args = parser.parse_args()

task = 'imdb'
model_checkpoint = "microsoft/deberta-v3-base"

model_name = model_checkpoint.split("/")[-1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load the IMDB dataset
dataset = load_dataset("imdb")

# Load the DeBERTa-v3-large model and tokenizer
model_name = "microsoft/deberta-v3-base"
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256) 

tokenized_datasets = dataset.map(tokenize_function, batched=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Load the accuracy metric
metric = evaluate.load("accuracy")

batch_size = 2
training_args = TrainingArguments(
    os.path.join(output_dir, f"{model_name}-finetuned-{task}"),
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


#small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10))
#small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(10))

# Instantiate Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Fine-tune the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()

# Print the evaluation results
print("Evaluation results:",  results["eval_accuracy"])

'''

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256) 

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

metric = evaluate.load("accuracy")
model_path = "/uufs/chpc.utah.edu/common/home/u1413911/local_exp/deberta_imdb/microsoft/deberta-v3-base-finetuned-imdb/checkpoint-12500" 
# Load the IMDb dataset
dataset = load_dataset("imdb")

# Load the fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
'''
# Instantiate a Trainer for evaluation
eval_trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics
)

# Evaluate the model
#train_results = eval_trainer.evaluate(tokenized_datasets["train"])
#print("Evaluation results:", train_results["eval_accuracy"])

# Evaluate the model
#evaluation_results = eval_trainer.evaluate(tokenized_datasets["test"])
#print("Evaluation results:", evaluation_results["eval_accuracy"])


PredictionOutput = eval_trainer.predict(tokenized_datasets["test"])
#predicted_labels = PredictionOutput.label_ids
predicted_labels = np.argmax(PredictionOutput.predictions, axis=1)
true_labels = tokenized_datasets["test"]["label"]

# Find indices of incorrectly predicted instances
incorrect_indices = np.where(predicted_labels != true_labels)[0]

# Randomly sample 10 incorrect instances
sampled_indices = np.random.choice(incorrect_indices, size=10, replace=False)
sampled_indices = sampled_indices.tolist()#[9733, 3259, 15116, 10140, 12304, 3740, 12498, 8166, 10975, 20502]

# Create a list to store sampled instances
output_items = []

# Iterate through sampled indices and collect instance information
for idx in sampled_indices:
    instance = {
        'review': tokenized_datasets["test"][idx]["text"],
        'label': int(true_labels[idx]),  # Convert int64 to Python int
        'predicted': int(predicted_labels[idx])  # Convert int64 to Python int
    }
    output_items.append(instance)

# Save sampled instances to a JSONLines file
filename  = "incorrect_predictions.jsonl"
with jsonlines.open(filename, mode='w') as writer:
    for item in output_items:
        writer.write(item)

print(f"Saved {len(output_items)} incorrect predictions to {filename}")
