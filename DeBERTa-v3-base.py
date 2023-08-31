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

output_dir = "./deberta_imdb"
parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Directory where model checkpoints will be saved')
args = parser.parse_args()

# your code

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
    num_train_epochs=5,
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
model_path = "./deberta_imdb/checkpoint-20835" 

# Load the IMDb dataset
dataset = load_dataset("imdb")

# Load the fine-tuned model
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Instantiate a Trainer for evaluation
eval_trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics
)

# Evaluate the model
evaluation_results = eval_trainer.evaluate(tokenized_datasets["train"])
print("Evaluation results:", evaluation_results["eval_accuracy"])

# Evaluate the model
evaluation_results = eval_trainer.evaluate(tokenized_datasets["test"])
print("Evaluation results:", evaluation_results["eval_accuracy"])
'''

