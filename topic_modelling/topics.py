#TOP_N_WORDS = 10
#MIN_TOPIC_SIZE = 200
N_TOPICS = 10
#MODEL = "mpnet"
VERBOSE = True

DATA_DIR = "/mnt/d/mag_data"

import datasets
import pickle
import torch
from matplotlib import pyplot as plt
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from collections import Counter
import os
from umap import UMAP

with open("temp_results", "rb") as file:
    data = pickle.load(file)

data = datasets.Dataset.from_dict(data)
data = data.sort("id")
data = data.filter(lambda x: x["length"]<1025)
texts = data[:]["input"]

# BEGIN EMBEDDINGS
if os.path.exists(os.path.join(DATA_DIR, "embeddings")):
    with open(os.path.join(DATA_DIR, "embeddings"), "rb") as file:
        embeddings_dict = pickle.load(file)

else:
    embeddings_dict = dict()

for MODEL in ["mpnet", "minilm"]:
    if MODEL in embeddings_dict.keys(): continue

    if MODEL == "mpnet":
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2").cuda()
    elif MODEL == "minilm":
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2").cuda()
    else:
        model = SentenceTransformer(MODEL).cuda()

    embeddings = model.encode(texts, show_progress_bar=VERBOSE)
    embeddings_dict[MODEL] = embeddings
    #data = data.add_column(name=f"{MODEL}_embedding", column=embeddings) # type: ignore
    del model
    torch.cuda.empty_cache()

    with open(os.path.join(DATA_DIR, "embeddings"), "wb") as file:
        pickle.dump(embeddings_dict, file)

# END EMBEDDINGS



for MODEL in ["mpnet", "minilm"]:
    if MODEL == "mpnet":
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2").cuda()
    elif MODEL == "minilm":
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2").cuda()
    else:
        model = SentenceTransformer(MODEL).cuda()


    embeddings_ = embeddings_dict[MODEL]
    if f"{MODEL}_reduced" not in embeddings_dict.keys():
        embeddings_dict[f"{MODEL}_reduced"] = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", verbose=VERBOSE).fit_transform(embeddings_)
        with open(os.path.join(DATA_DIR, "embeddings"), "wb") as file:
            pickle.dump(embeddings_dict, file)

    embeddings = embeddings_dict[f"{MODEL}_reduced"]


    class DimReduction:
        def fit(self, X):
            return self
        def transform(self, X):
            return embeddings

    dim_reduction = DimReduction()

    for TOP_N_WORDS in [5]:
        for MIN_TOPIC_SIZE in [1500]:
                params = f"{MODEL}-{MIN_TOPIC_SIZE}-{TOP_N_WORDS}-{N_TOPICS}"
                print(params)
                if os.path.exists(os.path.join(DATA_DIR, "topic_results", params)):
                    with open(os.path.join(DATA_DIR, "topic_results", params), "rb") as file:
                        data, topics_map, topics_dist = pickle.load(file)
                    print([a for _, a in topics_map.items()])
                    #print(topics_dist)
                    continue


                vectorizer = CountVectorizer(stop_words=["še","ter","zato","namreč","tem","niso","zaradi", "a","ali","april","avgust","b","bi","bil","bila","bile","bili","bilo","biti","blizu","bo","bodo","bojo","bolj","bom","bomo","boste","bova","boš","brez","c","cel","cela","celi","celo","d","da","daleč","dan","danes","datum","december","deset","deseta","deseti","deseto","devet","deveta","deveti","deveto","do","dober","dobra","dobri","dobro","dokler","dol","dolg","dolga","dolgi","dovolj","drug","druga","drugi","drugo","dva","dve","e","eden","en","ena","ene","eni","enkrat","eno","etc.","f","februar","g","g.","ga","ga.","gor","gospa","gospod","h","halo","i","idr.","ii","iii","in","iv","ix","iz","j","januar","jaz","je","ji","jih","jim","jo","julij","junij","jutri","k","kadarkoli","kaj","kajti","kako","kakor","kamor","kamorkoli","kar","karkoli","katerikoli","kdaj","kdo","kdorkoli","ker","ki","kje","kjer","kjerkoli","ko","koder","koderkoli","koga","komu","kot","kratek","kratka","kratke","kratki","l","lahka","lahke","lahki","lahko","le","lep","lepa","lepe","lepi","lepo","leto","m","maj","majhen","majhna","majhni","malce","malo","manj","marec","me","med","medtem","mene","mesec","mi","midva","midve","mnogo","moj","moja","moje","mora","morajo","moram","moramo","morate","moraš","morem","mu","n","na","nad","naj","najina","najino","najmanj","naju","največ","nam","narobe","nas","nato","nazaj","naš","naša","naše","ne","nedavno","nedelja","nek","neka","nekaj","nekatere","nekateri","nekatero","nekdo","neke","nekega","neki","nekje","neko","nekoga","nekoč","ni","nikamor","nikdar","nikjer","nikoli","nič","nje","njega","njegov","njegova","njegovo","njej","njemu","njen","njena","njeno","nji","njih","njihov","njihova","njihovo","njiju","njim","njo","njun","njuna","njuno","no","nocoj","november","npr.","o","ob","oba","obe","oboje","od","odprt","odprta","odprti","okoli","oktober","on","onadva","one","oni","onidve","osem","osma","osmi","osmo","oz.","p","pa","pet","peta","petek","peti","peto","po","pod","pogosto","poleg","poln","polna","polni","polno","ponavadi","ponedeljek","ponovno","potem","povsod","pozdravljen","pozdravljeni","prav","prava","prave","pravi","pravo","prazen","prazna","prazno","prbl.","precej","pred","prej","preko","pri","pribl.","približno","primer","pripravljen","pripravljena","pripravljeni","proti","prva","prvi","prvo","r","ravno","redko","res","reč","s","saj","sam","sama","same","sami","samo","se","sebe","sebi","sedaj","sedem","sedma","sedmi","sedmo","sem","september","seveda","si","sicer","skoraj","skozi","slab","smo","so","sobota","spet","sreda","srednja","srednji","sta","ste","stran","stvar","sva","t","ta","tak","taka","take","taki","tako","takoj","tam","te","tebe","tebi","tega","težak","težka","težki","težko","ti","tista","tiste","tisti","tisto","tj.","tja","to","toda","torek","tretja","tretje","tretji","tri","tu","tudi","tukaj","tvoj","tvoja","tvoje","u","v","vaju","vam","vas","vaš","vaša","vaše","ve","vedno","velik","velika","veliki","veliko","vendar","ves","več","vi","vidva","vii","viii","visok","visoka","visoke","visoki","vsa","vsaj","vsak","vsaka","vsakdo","vsake","vsaki","vsakomur","vse","vsega","vsi","vso","včasih","včeraj","x","z","za","zadaj","zadnji","zakaj","zaprta","zaprti","zaprto","zdaj","zelo","zunaj","č","če","često","četrta","četrtek","četrti","četrto","čez","čigav","š","šest","šesta","šesti","šesto","štiri","ž","že"])
                ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
                topic_model = BERTopic(umap_model=dim_reduction, verbose=VERBOSE, embedding_model=model, ctfidf_model=ctfidf_model, calculate_probabilities=False, top_n_words=TOP_N_WORDS, min_topic_size=MIN_TOPIC_SIZE, vectorizer_model=vectorizer) # type: ignore

                topics, probs = topic_model.fit_transform(texts, embeddings=embeddings_)
                data = data.add_column(name="topic", column=topics) # type: ignore
                topics_map = topic_model.get_topics()
                topics_dist = Counter(topics)
                with open(os.path.join(DATA_DIR, "topic_results", params), "wb") as file:
                    pickle.dump((data.select_columns(["id", "topic"]), topics_map, topics_dist), file)
                print(topics_map)
                print(topics_dist)
                data = data.remove_columns(["topic"])

                del vectorizer
                del ctfidf_model
                del topic_model
    del model
    torch.cuda.empty_cache()

