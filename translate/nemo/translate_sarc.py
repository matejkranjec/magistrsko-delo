from time import time
from re import findall

import torch
from nemo.collections.nlp.models import MTEncDecModel
from nemo.utils import logging
import contextlib
import csv
import json
import os

if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    autocast = torch.cuda.amp.autocast
else:
    @contextlib.contextmanager
    def autocast():
        yield

from nltk import download, sent_tokenize
download('punkt')

_TEXT_LEN_LIMIT = 5000
_TEXT_SPLIT_THRESHOLD = 1024
_SPLIT_LEN = 512
model = MTEncDecModel.restore_from("/d/hpc/projects/FRI/mk7972/models/aayn_base.nemo", map_location="cuda")


def translate_text(item):
  time0 = time()
  #logging.info(f" Q: {item}")

  if isinstance(item, str):
    text = [item]
  else:
    text = item
  text_len = sum(len(_text) for _text in text)
  if text_len > _TEXT_LEN_LIMIT:
    logging.warning(f'{text}, text length exceded {text_len}c [max {_TEXT_LEN_LIMIT}c]')

  text_batch = []
  text_batch_split = []
  for _text in text:
    if len(_text) > _TEXT_SPLIT_THRESHOLD:
      _split_start = len(text_batch)
      _sent = sent_tokenize(_text)
      i = 0
      while i < len(_sent):
        j = i+1
        while j < len(_sent) and len(' '.join(_sent[i:j])) < _SPLIT_LEN: j+=1
        if len(' '.join(_sent[i:j])) > _TEXT_SPLIT_THRESHOLD:
          _split=findall(rf'(.{{1,{_SPLIT_LEN}}})(?:\s|$)',' '.join(_sent[i:j]))
          text_batch.extend(_split)
        else:
          text_batch.append(' '.join(_sent[i:j]))
        i = j
      _split_end = len(text_batch)
      text_batch_split.append((_split_start,_split_end))
    else:
      text_batch.append(_text)

  #logging.debug(f' B: {text_batch}, BS: {text_batch_split}')


  translation_batch = model.translate(text_batch)
  #logging.debug(f' BT: {translation_batch}')

  translation = []
  _start = 0
  for _split_start,_split_end in text_batch_split:
    if _split_start != _start:
      translation.extend(translation_batch[_start:_split_start])
    translation.append(' '.join(translation_batch[_split_start:_split_end]))
    _start = _split_end
  if _start < len(translation_batch):
    translation.extend(translation_batch[_start:])


  #logging.debug(f'text_length: {text_len}c, duration: {round(time()-time0,2)}s')

  torch.cuda.empty_cache()
  return ' '.join(translation) if isinstance(text, str) else translation

import numpy as np
from itertools import groupby
from collections import defaultdict
from tqdm import tqdm


def process(anc, res, labels, filters):
    data = list()
    for a, r_, l_ in zip(anc, res, labels):
        for r, l in zip(r_, l_):
            data.append((a, r, l))
    
    for f in filters:
        data = list(filter(f, data))


    out = defaultdict(lambda: [list(), list()])
    for d in data:
        out[tuple(d[0])][0].append(d[1])
        out[tuple(d[0])][1].append(d[2])

    anc_, res_, lab_ = list(), list(), list()
    for o in out.items():
        anc_.append(list(o[0]))
        res_.append(o[1][0])
        lab_.append(o[1][1])

    return anc_, res_, lab_


def filter_length(x, min_length=20, max_length=200):
    return len(x[1])>min_length and len(x[1])<max_length and np.all([len(i)>min_length and len(i)<max_length for i in x[0]])

def filter_numbers(x, max_numbers=5):
    return sum([x[1].count(str(n)) for n in range(10)])<max_numbers and np.all([sum([i.count(str(n)) for n in range(10)])<max_numbers for i in x[0]])

def filter_chars(x, max_chars=1, chars="@#$%^_+={}|<>;"):
    return sum([x[1].count(n) for n in chars])<max_chars and np.all([sum([i.count(n) for n in chars])<max_chars for i in x[0]])




def filter_dataset(docs, labels, batch_size=2056, filters = [filter_length, filter_numbers, filter_chars]):
    n = len(labels)
    anc, res = docs["ancestors"], docs["responses"]
    anc_, res_, lab_ = list(), list(), list()
    for i in tqdm(range(0, n, batch_size)):
        a, r, l = process(anc[i:i+batch_size], res[i:i+batch_size], labels[i:i+batch_size], filters)
        anc_ += a
        res_ += r
        lab_ += l
    out = dict()
    out["ancestors"] = anc_
    out["responses"] = res_

    return out, lab_ 


import pickle
from tqdm import tqdm

with open("/d/hpc/projects/FRI/mk7972/data/english/train", "rb") as file:
    train = pickle.load(file)
with open("/d/hpc/projects/FRI/mk7972/data/english/test", "rb") as file:
    test = pickle.load(file)

with open("/d/hpc/projects/FRI/mk7972/data/english/train_labels", "rb") as file:
    train_labels = pickle.load(file)
with open("/d/hpc/projects/FRI/mk7972/data/english/test_labels", "rb") as file:
    test_labels = pickle.load(file)
    
train, train_labels = filter_dataset(train, train_labels)
test, test_labels = filter_dataset(test, test_labels)

anc, res = train["ancestors"], train["responses"]
anc += test["ancestors"]
res += test["responses"]
labels = train_labels + test_labels
anc = [" > ".join(a) for a in anc]

translated = list()

for i, a, r in zip(range(len(anc)), anc, res):
    translated.append(translate_text(" ; ".join([a, " ; ".join(r)])))
    if i%1000 == 0:
        with open("/d/hpc/projects/FRI/mk7972/data/slovene/sarc_filtered_translated", "wb") as file:
            pickle.dump(translated, file)
        with open("/d/hpc/projects/FRI/mk7972/data/slovene/sarc_filtered_labels", "wb") as file:
            pickle.dump(labels, file)


