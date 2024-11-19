import pickle
import datasets
import deepl

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
print(l)
for r in res2:
    l += sum([len(i) for i in r])
print(l)



inputs2 = list()
for a, r_ in zip(anc2, res2):
    for r in r_:
        inputs2.append("translate to slovene: "+a+" "+r)
text2 = ["translate to slovene: "+d for d in dataset_onion["eng"]]
inputs2 += text2
print(inputs2[:10], inputs2[-10:])

for t in text2:
    l += len(t)

print(l)

dataset = datasets.Dataset.from_dict({"slo":inputs, "eng":inputs2, "label":labels, "prob":prob})
dataset.sort(column_names="prob")
dataset = dataset.remove_columns(["prob"])
dataset = dataset.select(range(len(inputs)-100000, len(inputs))).shuffle(seed=42)
bad, acceptable, good, sarcasm = list(), list(), list(), list()

l_ = 0
for d in dataset:
    l_ += len(d["eng"])
print("l_ ", l_)
with open("data_eval_results", "rb") as file:
    bad, acceptable, good, sarcasm = pickle.load(file)


for i in range(len(bad)+len(acceptable)+len(good), 1000):
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
    with open("data_eval_results", "wb") as file:
        pickle.dump((bad, acceptable, good, sarcasm), file)

ALL = len(bad)+len(acceptable)+len(good)
print(f"SLABI: {len(bad)} ({len(bad)/ALL}), SPREJEMLJIVI: {len(acceptable)} ({len(acceptable)/ALL}), DOBRI: {len(good)} ({len(good)/ALL}), ALL: {ALL}")
print(f"SARKAZEM OHRANJEN: {sum(sarcasm)} ({sum(sarcasm)/len(sarcasm)}), VSEH SARKASTICNIH: {len(sarcasm)}")

exit(0)
#deepl_translations = dict()
#deepl_translations["eng"] = [b[0] for b in bad]
#deepl_translations["slo"] = list()

# API_KEY = "3c84a946-35f0-4894-81c0-40d9052801fb:fx"
# translator = deepl.Translator(API_KEY)


#for e in deepl_translations["eng"][:100]:
#    deepl_translations["slo"].append(translator.translate_text(e, target_lang="SL").text)

#with open("data/deepl/test", "wb") as f:
#    pickle.dump(deepl_translations, f)

# print(deepl_translations["eng"][:5])
# print(deepl_translations["slo"])
# exit(0)

with open("data/deepl/test", "rb") as f:
    deepl_translations = pickle.load(f)


bad,acceptable,good = list(), list(), list()
for i in range(len(deepl_translations["slo"])):
    eng = deepl_translations["eng"][i]
    slo = deepl_translations["slo"][i]

    print(f"ENG: {eng}\nSLO:{slo}\n")
    c = int(input("prevod (0: SLAB, 1: SPREJEMLJIV, 2: DOBER): "))
    if c==0: bad.append((eng,slo))
    if c==1: acceptable.append((eng,slo))
    if c==2: good.append((eng,slo))

print(f"SLABI: {len(bad)} ({len(bad)/ALL}), SPREJEMLJIVI: {len(acceptable)} ({len(acceptable)/ALL}), DOBRI: {len(good)} ({len(good)/ALL}), ALL: {ALL}")

deepl_translations["eval"] = (bad, acceptable, good)
with open("data/deepl/test", "wb") as f:
    pickle.dump(deepl_translations, f)
