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
model = MTEncDecModel.restore_from("/d/hpc/projects/FRI/DL/mk7972/models/aayn_base.nemo", map_location="cuda")


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



import pickle
from tqdm import tqdm


docs, labels = list(), list()

with open("/d/hpc/projects/FRI/DL/mk7972/data/onion_data", "rb") as file:
    data = pickle.load(file)

print("begin translating")
for i in data:
  docs.append(translate_text(i["input"]))
  labels.append(i["label"])


with open("/d/hpc/projects/FRI/DL/mk7972/data/onion_translated", "wb") as file:
    pickle.dump(docs, file)

with open("/d/hpc/projects/FRI/DL/mk7972/data/onion_labels", "wb") as file:
    pickle.dump(labels, file)

