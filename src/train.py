from src.preprocess import readfile, add_chars, create_matrices, padding, create_batches, get_words_and_labels, iterate_mini_batches
from embedding.embedding import get_word_embedding, get_case_embedding, get_char_index_matrix, get_label_index_matrix, get_pos_tag_embedding
from src.model import get_model
from src.validation import compute_f1
from keras.utils import Progbar
import numpy as np


def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i, data in enumerate(dataset):
        tokens, casing, char, labels, pos_tag = data
        tokens = np.asarray([tokens])
        casing = np.asarray([casing])
        char = np.asarray([char])
        pos_tag = np.asarray([pos_tag])
        pred = model.predict([tokens, pos_tag, casing, char], verbose=False)[0]
        pred = pred.argmax(axis=-1)  # Predict the classes
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels


train = readfile("../data/train.txt")
validation = readfile("../data/valid.txt")
test = readfile("../data/test.txt")

train = add_chars(train)
validation = add_chars(validation)
test = add_chars(test)

words, labelSet, pos_tag_set = get_words_and_labels(train, validation, test)
label_index = get_label_index_matrix(labelSet)
pos_tag_index, posTagEmbedding = get_pos_tag_embedding(pos_tag_set)
case_index, caseEmbeddings = get_case_embedding()
word_index, wordEmbeddings = get_word_embedding(words)
char_index = get_char_index_matrix()

train_set = padding(create_matrices(train, word_index,  label_index, case_index, char_index, pos_tag_index))
validation_set = padding(create_matrices(validation, word_index, label_index, case_index, char_index, pos_tag_index))
test_set = padding(create_matrices(test, word_index, label_index, case_index, char_index, pos_tag_index))

idx2Label = {v: k for k, v in label_index.items()}

train_batch, train_batch_len = create_batches(train_set)
dev_batch, dev_batch_len = create_batches(validation_set)
test_batch, test_batch_len = create_batches(test_set)

model = get_model(wordEmbeddings, caseEmbeddings, char_index, label_index, posTagEmbedding)

epochs = 80
for epoch in range(epochs):
    print("Epoch %d/%d"%(epoch, epochs))
    for i, batch in enumerate(iterate_mini_batches(train_batch, train_batch_len)):
        labels, tokens, casing, char, pos_tag = batch
        model.train_on_batch([tokens, pos_tag, casing, char], labels)
    #   Performance on dev dataset
    predLabels, correctLabels = tag_dataset(dev_batch)
    pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, idx2Label)
    print("Dev-Data: Prec: %.5f, Rec: %.5f, F1: %.5f" % (pre_dev, rec_dev, f1_dev))

#   Performance on test dataset
predLabels, correctLabels = tag_dataset(test_batch)
pre_test, rec_test, f1_test = compute_f1(predLabels, correctLabels, idx2Label)
print("Test-Data: Prec: %.5f, Rec: %.5f, F1: %.5f" % (pre_test, rec_test, f1_test))
