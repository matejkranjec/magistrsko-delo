import pickle
import os
import datasets



if not os.path.exists('sample_dataset'):
    with open('sample', 'rb') as file:
        sample = pickle.load(file)



    with open("temp_results", "rb") as file:
        data = pickle.load(file)

    data = datasets.Dataset.from_dict(data).filter(lambda x: x['length']<1025)


    def map_(x):
        for i in range(20):
            if x['id'] in sample[i]: return {'topic':i+1}
        return {'topic':0}

    data = data.map(map_).filter(lambda x: not x['topic'] == 0)

    with open('sample_dataset', 'wb') as file:
        pickle.dump(data, file)
else:
    with open('sample_dataset', 'rb') as file:
        data = pickle.load(file)



for topic in range(5): 
    data_ = data.filter(lambda x: x['topic'] == topic)
    sarcasm = 0
    all = 0
    for d in data_:
        print(d['input'], '\n') # type: ignore
        s_ = int(input('0: non-sarcastic, 1: sarcastic\n'))
        if (s_ > 1): break
        sarcasm += s_
        all += 1
        print('-----------------------------------------------------------------------------------------------------------------------------------\n')
    print(sarcasm, all, '\n')

