from src.preprocess import readfile, add_chars, create_matrices, padding, create_batches, get_words_and_labels, iterate_mini_batches
from embedding.embedding import get_word_embedding, get_case_embedding, get_char_index_matrix, get_label_index_matrix
from src.model import get_model
from src.validation import compute_f1
from keras.utils import Progbar
import numpy as np


def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i, data in enumerate(dataset):
        tokens, casing, char, labels = data
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = model.predict([tokens, casing,char], verbose=False)[0]
        pred = pred.argmax(axis=-1)  # Predict the classes
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels


train = readfile("data/train.txt")
validation = readfile("data/valid.txt")
test = readfile("data/test.txt")

train = add_chars(train)
validation = add_chars(validation)
test = add_chars(test)

words, labelSet = get_words_and_labels(train, validation, test)
label_index = get_label_index_matrix(labelSet)
case_index, caseEmbeddings = get_case_embedding()
word_index, wordEmbeddings = get_word_embedding(words)
char_index = get_char_index_matrix()

train_set = padding(create_matrices(train, word_index,  label_index, case_index, char_index))
validation_set = padding(create_matrices(validation, word_index, label_index, case_index, char_index))
test_set = padding(create_matrices(test, word_index, label_index, case_index))

idx2Label = {v: k for k, v in label_index.items()}

train_batch, train_batch_len = create_batches(train_set)
dev_batch, dev_batch_len = create_batches(validation_set)
test_batch, test_batch_len = create_batches(test_set)

model = get_model(wordEmbeddings,caseEmbeddings,char_index, label_index)

epochs = 50
for epoch in range(epochs):
    print("Epoch %d/%d"%(epoch,epochs))
    a = Progbar(len(train_batch_len))
    for i,batch in enumerate(iterate_mini_batches(train_batch,train_batch_len)):
        labels, tokens, casing,char = batch
        model.train_on_batch([tokens, casing,char], labels)
        a.update(i)
        print(' ')

#   Performance on dev dataset
predLabels, correctLabels = tag_dataset(dev_batch)
pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, idx2Label)
print("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev))

#   Performance on test dataset
predLabels, correctLabels = tag_dataset(test_batch)
pre_test, rec_test, f1_test = compute_f1(predLabels, correctLabels, idx2Label)
print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))
