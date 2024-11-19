
ARTICLES_DIR = "/d/hpc/home/mk7972/articles/articles2021"
CACHE_DIR = "/d/hpc/projects/FRI/mk7972/huggingface"
FINAL_MODELS_DIR = "/d/hpc/projects/FRI/mk7972/models"
MODEL_NAME = "sloberta-sarcasm"
CHECKPOINT_NAME = "embeddia/sloberta"
TRAINERS_DIR = "/d/hpc/projects/FRI/mk7972/trainers"

import jsonlines
import os
from tqdm import tqdm, trange
files = os.listdir(ARTICLES_DIR)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


texts = list()
uri = list()
filenames = list()

for file in tqdm(files):
    with jsonlines.open(os.path.join(ARTICLES_DIR, file)) as f:
        for line in f.iter():
            uri.append(line["uri"]) 
            texts.append(line["body"])
            filenames.append(file)

ARTICLES_DIR = "/d/hpc/home/mk7972/articles/articles2022"
files = os.listdir(ARTICLES_DIR)

for file in tqdm(files):
    with jsonlines.open(os.path.join(ARTICLES_DIR, file)) as f:
        for line in f.iter():
            uri.append(line["uri"]) 
            texts.append(line["body"])
            filenames.append(file)

data_ = {"input":texts, "uri":uri, "file":filenames}

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(FINAL_MODELS_DIR, MODEL_NAME), cache_dir=CACHE_DIR)
from transformers import Trainer, TrainingArguments
import datasets
args = TrainingArguments(output_dir=os.path.join(TRAINERS_DIR, MODEL_NAME), logging_strategy="steps", logging_steps=1)#, eval_accumulation_steps=32)


trainer = Trainer(model=model, args=args,)
data = datasets.Dataset.from_dict({"input":texts})
import torch
MAX_LENGTH=512


def preprocess(x):
    return tokenizer(x["input"], max_length=MAX_LENGTH, truncation=True, padding="max_length")

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_NAME, cache_dir=CACHE_DIR)


data = data.map(preprocess, batched=True, batch_size=250)


preds = trainer.predict(test_dataset=data)





probs = torch.softmax(torch.tensor(preds.predictions), dim=1)
labels = torch.argmax(probs, dim=1)
data = data[:]
data["probs"] = probs
data["labels"] = labels




import pickle

with open("/d/hpc/home/mk7972/articles/results-sloberta", "wb") as file:
    pickle.dump(data, file)