CACHE_DIR = "/d/hpc/projects/FRI/mk7972/huggingface"
DATA_DIR = "/d/hpc/projects/FRI/mk7972/data"
TRAINERS_DIR = "/d/hpc/projects/FRI/mk7972/trainers"
FINAL_MODELS_DIR = "/d/hpc/projects/FRI/mk7972/models"

CHECKPOINT_NAME = "cjvt/t5-sl-large"
MODEL_NAME = "slot5-sarcasm"
GRADIENT_ACCUMULATION_STEPS = 4
EARLY_STOPPING_PATIENCE = 4

import pickle
import datasets
import os
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, EarlyStoppingCallback
import evaluate
from transformers import AutoTokenizer

os.environ["WANDB__SERVICE_WAIT"]="300"

datasets.disable_progress_bar()

with open(os.path.join(DATA_DIR, "combined-final"), "rb") as file:
    d = pickle.load(file)
dataset = datasets.Dataset.from_dict(d)
dataset = dataset.rename_column("slo", "input")

start_val = int(0.7*len(dataset))
start_test = int(0.85*len(dataset))

data_train = datasets.Dataset.from_dict(dataset[:start_val])
data_val = datasets.Dataset.from_dict(dataset[start_val:start_test])
data_test = datasets.Dataset.from_dict(dataset[start_test:])

########################


MAX_LENGTH=256
def preprocess(x):
    return tokenizer(x["input"], max_length=MAX_LENGTH, padding="max_length", truncation=True)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_NAME, cache_dir=CACHE_DIR)

tokenized_train = data_train.map(preprocess, batched=True)
tokenized_val = data_val.map(preprocess, batched=True)
tokenized_test = data_test.map(preprocess, batched=True)

tokenized_test = tokenized_test.remove_columns(column_names=["input"])
tokenized_train = tokenized_train.remove_columns(column_names=["input"])
tokenized_val = tokenized_val.remove_columns(column_names=["input"])

os.environ["WANDB_PROJECT"]="mag"

model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_NAME, cache_dir=CACHE_DIR)


args = TrainingArguments(learning_rate=1e-5, gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS, disable_tqdm=True, load_best_model_at_end=True, output_dir=os.path.join(TRAINERS_DIR, MODEL_NAME), save_strategy="epoch", evaluation_strategy="epoch", num_train_epochs=20, logging_steps=1000/GRADIENT_ACCUMULATION_STEPS, report_to="wandb", run_name=MODEL_NAME)#, eval_accumulation_steps=32)
#args = TrainingArguments(output_dir="trainer", save_strategy="epoch", evaluation_strategy="steps", eval_steps=100, num_train_epochs=5, logging_steps=1000)

accuracy = evaluate.load("accuracy")

def ca(pred):
    return accuracy.compute(predictions=pred.predictions[0], references=pred.predictions[1])

import torch

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits[0], axis=1)
    return pred_ids, labels

trainer = Trainer(model=model, args=args, train_dataset=tokenized_train, eval_dataset=tokenized_val, compute_metrics=ca, preprocess_logits_for_metrics=preprocess_logits_for_metrics)
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=2))
trainer.train()

trainer.evaluate(eval_dataset=tokenized_test)
trainer.save_model(os.path.join(FINAL_MODELS_DIR, MODEL_NAME))

