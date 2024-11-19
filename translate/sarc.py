import os
import csv
import json
import pickle

def load_sarc_responses(dataset_path, train_file_="train-balanced.csv", test_file_="test-balanced.csv", comment_file_="comments.json", lower=False):
  '''loads SARC data from csv files
  Args:
    train_file: csv file with train sequences
    test_file: csv file with train sequences
    comment_file: json file with details about all comments
    lower: boolean; if True, converts comments to lowercase
  Returns:
    train_sequences, train_labels, test_sequences, test_labels
    train_sequences: {'ancestors': list of ancestors for all sequences,
                      'responses': list of responses for all sequences}
    train_labels: list of labels for responses for all sequences.
  '''
  train_file = os.path.join(dataset_path, train_file_)
  test_file = os.path.join(dataset_path, test_file_)
  comment_file = os.path.join(dataset_path, comment_file_)

  with open(comment_file, 'r') as f:
    comments = json.load(f)

  train_docs = {'ancestors': [], 'responses': []}
  train_labels = []
  with open(train_file, 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
      ancestors = row[0].split(' ')
      responses = row[1].split(' ')
      labels = row[2].split(' ')
      if lower:
        train_docs['ancestors'].append([comments[r]['text'].lower() for r in ancestors])
        train_docs['responses'].append([comments[r]['text'].lower() for r in responses])
      else:
        train_docs['ancestors'].append([comments[r]['text'] for r in ancestors])
        train_docs['responses'].append([comments[r]['text'] for r in responses])
      train_labels.append(labels)
  

  test_docs = {'ancestors': [], 'responses': []}
  test_labels = []
  with open(test_file, 'r') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
      ancestors = row[0].split(' ')
      responses = row[1].split(' ')
      labels = row[2].split(' ')
      if lower:
        test_docs['ancestors'].append([comments[r]['text'].lower() for r in ancestors])
        test_docs['responses'].append([comments[r]['text'].lower() for r in responses])
      else:
        test_docs['ancestors'].append([comments[r]['text'] for r in ancestors])
        test_docs['responses'].append([comments[r]['text'] for r in responses])
      test_labels.append(labels)

  return train_docs, test_docs, train_labels, test_labels


if __name__ == "__main__":
  #train_docs, test_docs, train_labels, test_labels = load_sarc_responses("data/main", train_file_="train-unbalanced.csv", test_file_="test-unbalanced.csv")
  train_docs, test_docs, train_labels, test_labels = load_sarc_responses("data/main")
  print(len(test_labels))


  exit(0)
  with open("data/train_unbalanced", "wb") as file:
    pickle.dump(train_docs, file)
  with open("data/test_unbalanced", "wb") as file:
    pickle.dump(test_docs, file)
  with open("data/train_unbalanced_labels", "wb") as file:
    pickle.dump(train_labels, file)
  with open("data/test_unbalanced_labels", "wb") as file:
    pickle.dump(test_labels, file)