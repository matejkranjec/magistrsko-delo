import os
import csv
import json
import pickle

def load_onion(dataset_path, file_="Sarcasm_Headlines_Dataset_v2.json", lower=True):
  file = os.path.join(dataset_path, file_)

  with open(file, 'r') as f:
    data = json.load(f)

  data = list(map(lambda x: {"label": x["is_sarcastic"], "input":x["headline"]}, data))
  return data


if __name__ == "__main__":
  data = load_onion("data/onion")
  with open("data/onion_data", "wb") as file:
    pickle.dump(data, file)
