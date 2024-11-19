
ARTICLES_DIR = "/d/hpc/projects/FRI/mk7972/articles"
CACHE_DIR = "/d/hpc/projects/FRI/mk7972/huggingface"
FINAL_MODELS_DIR = "/d/hpc/projects/FRI/mk7972/models"
CHECKPOINT_NAME = "meta-llama/Meta-Llama-3-8B"
MODEL_NAME = "llama3-sarcasm"
TRAINERS_DIR = "/d/hpc/projects/FRI/mk7972/trainers"

PART = 2

import jsonlines
import os
from tqdm import tqdm, trange
files = os.listdir(ARTICLES_DIR)




from peft import LoraConfig, TaskType, get_peft_model, AutoPeftModelForSequenceClassification
from transformers import AutoModelForSequenceClassification


peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=True, r=16, lora_alpha=32, lora_dropout=0.1, bias="lora_only")



model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(FINAL_MODELS_DIR, MODEL_NAME), cache_dir=CACHE_DIR)
model = get_peft_model(model, peft_config)




from transformers import Trainer, TrainingArguments, AutoTokenizer
import datasets
args = TrainingArguments(output_dir=os.path.join(TRAINERS_DIR, MODEL_NAME), logging_strategy="steps", logging_steps=1000)#, eval_accumulation_steps=32)
datasets.disable_progress_bar()


trainer = Trainer(model=model, args=args)
import pickle
with open(os.path.join(ARTICLES_DIR, f"articles_part{PART}"), "rb") as file:
    data = datasets.Dataset.from_dict(pickle.load(file))


import torch

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_NAME, cache_dir=CACHE_DIR)

model.config.pad_token_id = tokenizer.eos_token_id





preds = trainer.predict(test_dataset=data)



probs = torch.softmax(torch.tensor(preds.predictions), dim=1)
labels = torch.argmax(probs, dim=1)
data = data[:]
data["probs"] = probs
data["labels"] = labels




import pickle

with open(os.path.join(ARTICLES_DIR, f"results_part{PART}"), "wb") as file:
    pickle.dump(data, file)