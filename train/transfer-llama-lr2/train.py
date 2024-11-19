CACHE_DIR = "/d/hpc/projects/FRI/mk7972/huggingface"
DATA_DIR = "/d/hpc/projects/FRI/mk7972/data"
TRAINERS_DIR = "/d/hpc/projects/FRI/mk7972/trainers"
FINAL_MODELS_DIR = "/d/hpc/projects/FRI/mk7972/models"

CHECKPOINT_NAME = "meta-llama/Meta-Llama-3-8B"
MODEL_NAME = "llama-sarcasm-transfer-lr2"
GRADIENT_ACCUMULATION_STEPS = 4
EARLY_STOPPING_PATIENCE = 4

import pickle
import datasets
import os
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, EarlyStoppingCallback, BitsAndBytesConfig
import evaluate
from transformers import AutoTokenizer

os.environ["WANDB__SERVICE_WAIT"]="300"

datasets.disable_progress_bar()

with open(os.path.join(DATA_DIR, "combined-final"), "rb") as file:
    d = pickle.load(file)
dataset = datasets.Dataset.from_dict(d)


start_val = int(0.7*len(dataset))
start_test = int(0.85*len(dataset))

data_train = datasets.Dataset.from_dict(dataset[:start_val])
data_train = data_train.rename_column("eng", "input")

data_val = datasets.Dataset.from_dict(dataset[start_val:start_test])
data_val = data_val.rename_column("slo", "input")

data_test = datasets.Dataset.from_dict(dataset[start_test:])
data_test = data_test.rename_column("slo", "input")
########################


MAX_LENGTH=256
def preprocess(x):
    return tokenizer(x["input"], max_length=MAX_LENGTH, padding="max_length", truncation=True)

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_NAME, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token

tokenized_train = data_train.map(preprocess, batched=True)
tokenized_val = data_val.map(preprocess, batched=True)
tokenized_test = data_test.map(preprocess, batched=True)

tokenized_test = tokenized_test.remove_columns(column_names=["input"])
tokenized_train = tokenized_train.remove_columns(column_names=["input"])
tokenized_val = tokenized_val.remove_columns(column_names=["input"])

os.environ["WANDB_PROJECT"]="mag"

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_NAME, cache_dir=CACHE_DIR, quantization_config=quantization_config)
model.config.pad_token_id = tokenizer.pad_token_id


from peft import LoraConfig, TaskType, get_peft_model

peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1, bias="lora_only")

model = get_peft_model(model, peft_config)


args = TrainingArguments(learning_rate=1e-5, optim="adamw_bnb_8bit", gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS, disable_tqdm=True, load_best_model_at_end=True, output_dir=os.path.join(TRAINERS_DIR, MODEL_NAME), save_strategy="epoch", evaluation_strategy="epoch", num_train_epochs=20, logging_steps=1000/GRADIENT_ACCUMULATION_STEPS, report_to="wandb", run_name=MODEL_NAME)#, eval_accumulation_steps=32)
#args = TrainingArguments(output_dir="trainer", save_strategy="epoch", evaluation_strategy="steps", eval_steps=100, num_train_epochs=5, logging_steps=1000)

accuracy = evaluate.load("accuracy")

def ca(pred):
    return accuracy.compute(predictions=pred.predictions[0], references=pred.predictions[1])

import torch

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, axis=1)
    return pred_ids, labels

trainer = Trainer(model=model, args=args, train_dataset=tokenized_train, eval_dataset=tokenized_val, compute_metrics=ca, preprocess_logits_for_metrics=preprocess_logits_for_metrics)
trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))

trainer.train()

trainer.evaluate(eval_dataset=tokenized_test)
trainer.save_model(os.path.join(FINAL_MODELS_DIR, MODEL_NAME))

