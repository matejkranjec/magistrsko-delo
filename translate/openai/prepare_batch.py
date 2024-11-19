import pickle
import datasets

with open("data/parallel/SARC_PARALLEL", "rb") as file:
    dataset_sarc = pickle.load(file)
with open("data/parallel/ONION_PARALLEL", "rb") as file:
    dataset_onion = pickle.load(file)


anc, res, lab, prob_ = dataset_sarc["slo_anc"], dataset_sarc["slo_res"], dataset_sarc["labels"], dataset_sarc["prob"]
inputs, labels, prob = list(), list(), list()
for a, r_, l_, p in zip(anc, res, lab, prob_):
    for r, l in zip(r_, l_):
        inputs.append(a+r)
        labels.append(l)
        prob.append(p)

text, lab, prob_ = dataset_onion["slo"], dataset_onion["labels"], dataset_onion["prob"]
inputs += text
labels += lab
prob += prob_

anc2, res2 = dataset_sarc["eng_anc"], dataset_sarc["eng_res"]

l = 0
for a in anc2:
    l+=len(a)

for r in res2:
    l += sum([len(i) for i in r])




inputs2 = list()
for a, r_ in zip(anc2, res2):
    for r in r_:
        inputs2.append(a+"; "+r)
text2 =dataset_onion["eng"]
inputs2 += text2

for t in text2:
    l += len(t)



dataset = datasets.Dataset.from_dict({"slo":inputs, "eng":inputs2, "label":labels, "prob":prob})
dataset.sort(column_names="prob")
dataset = dataset.remove_columns(["prob"])
dataset = dataset.select(range(len(inputs)-100000, len(inputs))).shuffle(seed=42)


from tqdm import tqdm
from multiprocessing import Pool

def f(i):
    return {"custom_id": f"{i}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "system", "content": "Translate to Slovene."}, {"role": "user", "content": dataset["eng"][i]}],"max_tokens": 1000}}


N = 100000
if __name__ == "__main__":
    with Pool(14) as p:
        queries = list(tqdm(p.imap(f, range(N)), total=N))
    import pickle
    with open("queries", "wb") as file:
        pickle.dump(queries, file)
    import json
    with open('output.jsonl', 'w') as outfile:
        for entry in queries:
            json.dump(entry, outfile)
            outfile.write('\n')
