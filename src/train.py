from src.preprocess import readfile, add_chars, create_matrices, create_batches, get_words_and_labels, transform
from embedding.embedding import get_word_embedding, get_case_embedding, get_char_index_matrix, get_label_index_matrix, get_pos_tag_embedding
from src.model import get_model
from src.validation import compute_f1
from keras.utils import Progbar
import numpy as np
from sklearn import metrics
from itertools import chain


def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i, data in enumerate(dataset):
        tokens, casing, char, labels, pos_tag = data
        input, output = transform([[tokens, casing, char, labels, pos_tag]], max(2,len(labels)))
        pred = model.predict(input, verbose=False)[0]
        pred = pred.argmax(axis=-1)  # Predict the classes
        output = np.squeeze(output)
        output = np.argmax(output, axis=1)

        if output.shape[0] == 2 and output[1] == 0:
            output = np.delete(output, [1])
            pred = np.delete(pred, [1])

        correctLabels.append(output)
        predLabels.append(pred)
        b.update(i)

    print(metrics.classification_report(list(chain.from_iterable(correctLabels)), list(chain.from_iterable(predLabels))))
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

train_set = create_matrices(train, word_index,  label_index, case_index, char_index, pos_tag_index)
validation_set = create_matrices(validation, word_index, label_index, case_index, char_index, pos_tag_index)
test_set = create_matrices(test, word_index, label_index, case_index, char_index, pos_tag_index)

batch_size =20
model = get_model(wordEmbeddings, caseEmbeddings, char_index, posTagEmbedding, batch_size)

train_steps, train_batches = create_batches(train_set, batch_size)

epochs = 10
model.fit_generator(generator=train_batches, steps_per_epoch=train_steps, epochs=epochs)

idx2Label = {v: k for k, v in label_index.items()}
#   Performance on dev dataset
predLabels, correctLabels = tag_dataset(validation_set)
pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, idx2Label)
print("Dev-Data: Prec: %.5f, Rec: %.5f, F1: %.5f" % (pre_dev, rec_dev, f1_dev))

#   Performance on test dataset
predLabels, correctLabels = tag_dataset(test_set)
pre_test, rec_test, f1_test = compute_f1(predLabels, correctLabels, idx2Label)
print("Test-Data: Prec: %.5f, Rec: %.5f, F1: %.5f" % (pre_test, rec_test, f1_test))