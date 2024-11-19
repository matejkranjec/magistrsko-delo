import pickle
import datasets
import os
from tqdm import trange

with open("data/parallel/combined-final", "rb") as file:
    dataset = pickle.load(file)


dataset = datasets.Dataset.from_dict(dataset)
bad, acceptable, good, sarcasm = list(), list(), list(), list()

l_ = 0
for d in dataset:
    l_ += len(d["eng"])

if os.path.exists('data_eval_results_gpt'):
    with open("data_eval_results_gpt", "rb") as file:
        bad, acceptable, good, sarcasm = pickle.load(file)


for i in trange(len(bad)+len(acceptable)+len(good), 1000):
    eng = dataset[i]["eng"]
    slo = dataset[i]["slo"]
    label = dataset[i]["label"]
    print(f"ENG: {eng}\nSLO:{slo}\n")
    c = int(input("prevod (0: SLAB, 1: SPREJEMLJIV, 2: DOBER): "))
    if c==0: bad.append((eng,slo))
    if c==1: acceptable.append((eng,slo))
    if c==2: good.append((eng,slo))

    if c>0:
        if label:
            s = int(input("ohranjen sarkazem (DA: 1, NE: 0): "))
            sarcasm.append(s)
    print("\n")
    with open("data_eval_results_gpt", "wb") as file:
        pickle.dump((bad, acceptable, good, sarcasm), file)

ALL = len(bad)+len(acceptable)+len(good)
print(f"SLABI: {len(bad)} ({len(bad)/ALL}), SPREJEMLJIVI: {len(acceptable)} ({len(acceptable)/ALL}), DOBRI: {len(good)} ({len(good)/ALL}), ALL: {ALL}")
print(f"SARKAZEM OHRANJEN: {sum(sarcasm)} ({sum(sarcasm)/len(sarcasm)}), VSEH SARKASTICNIH: {len(sarcasm)}")

exit(0)
'''
SLABI: 10 (0.01), SPREJEMLJIVI: 59 (0.059), DOBRI: 931 (0.931), ALL: 1000
SARKAZEM OHRANJEN: 495 (0.99), VSEH SARKASTICNIH: 500
'''