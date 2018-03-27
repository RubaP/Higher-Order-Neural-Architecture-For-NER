import numpy as np


def get_word_embedding(words):
    word_index = {}
    word_embeddings = []
    embeddings = open("../embedding/glove.6B.100d.txt", encoding="utf-8")

    for line in embeddings:
        split = line.strip().split(" ")

        if len(word_index) == 0:  # Add padding+unknown
            word_index["PADDING_TOKEN"] = len(word_index)
            vector = np.zeros(len(split) - 1)  # Zero vector vor 'PADDING' word
            word_embeddings.append(vector)

            word_index["UNKNOWN_TOKEN"] = len(word_index)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
            word_embeddings.append(vector)

        if split[0].lower() in words:
            vector = np.array([float(num) for num in split[1:]])
            word_embeddings.append(vector)
            word_index[split[0]] = len(word_index)

    word_embeddings = np.array(word_embeddings)
    return word_index, word_embeddings


def get_case_embedding():
    # Hard coded case lookup
    case_index = {'numeric': 7, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                  'contains_digit': 6, 'PADDING_TOKEN': 0}
    embedding = np.identity(len(case_index), dtype='float32')
    return case_index, embedding


def get_pos_tag_embedding(pos_tag_set):
    # Hard coded case lookup
    pos_tag_index = get_POS_tag_index_matrix(pos_tag_set)
    embedding = np.identity(len(pos_tag_index), dtype='float32')
    return pos_tag_index, embedding


def get_char_index_matrix():
    char_index = {"PADDING": 0, "UNKNOWN": 1}
    for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
        char_index[c] = len(char_index)
    return char_index


def get_label_index_matrix(label_set):
    label_index = {'PAD': 0, 'O\n': 1, 'B-PER\n': 2, 'I-PER\n': 3, 'B-LOC\n': 4, 'I-LOC\n': 5, 'B-ORG\n': 6, 'I-ORG\n': 7,
                   'B-MISC\n': 8, 'I-MISC\n': 9}
    return label_index


def get_POS_tag_index_matrix(POS_tag_set):
    POS_tag_index = {"PAD":0}
    for POS_tag in POS_tag_set:
        POS_tag_index[POS_tag] = len(POS_tag_index)
    return POS_tag_index
