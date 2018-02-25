import numpy as np
from keras.models import Model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D,\
    Flatten, concatenate
from keras.utils import plot_model, Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
from preprocessing.preprocess import readfile, add_chars, create_matrices, padding

train = readfile("data/train.txt")
validation = readfile("data/valid.txt")
test = readfile("data/test.txt")

train = add_chars(train)
validation = add_chars(validation)
test = add_chars(test)

labelSet = set()
words = {}

for dataset in [train, validation, test]:
    for sentence in dataset:
        for word, char, label in sentence:
            labelSet.add(label)
            words[word.lower()] = True

# Create mapping for the labels
label_index = {}
for label in labelSet:
    label_index[label] = len(label_index)

# Hard coded case lookup
case_index = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
              'contains_digit': 6, 'PADDING_TOKEN': 7}
caseEmbeddings = np.identity(len(case_index), dtype='float32')

word_index = {}
wordEmbeddings = []
embeddings = open("embeddings/glove.6B.100d.txt", encoding="utf-8")

for line in embeddings:
    split = line.strip().split(" ")
    word = split[0]

    if len(word_index) == 0:  # Add padding+unknown
        word_index["PADDING_TOKEN"] = len(word_index)
        vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)

        word_index["UNKNOWN_TOKEN"] = len(word_index)
        vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
        wordEmbeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word_index[split[0]] = len(word_index)

wordEmbeddings = np.array(wordEmbeddings)

char_index = {"PADDING": 0, "UNKNOWN": 1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char_index[c] = len(char_index)

train_set = padding(create_matrices(train, word_index,  label_index, case_index, char_index))
validation_set = padding(create_matrices(validation, word_index, label_index, case_index, char_index))
test_set = padding(create_matrices(test, word_index, label_index,case_index))

idx2Label = {v: k for k, v in label_index.items()}

train_batch, train_batch_len = create_matrices(train_set)
dev_batch, dev_batch_len = create_matrices(validation_set)
test_batch, test_batch_len = create_matrices(test_set)
