import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences


def readfile(filename):
    '''
        read file
        return format :
        [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O']]
    '''
    f = open(filename)
    sentences = []
    sentence = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split(' ')
        sentence.append([splits[0], splits[-1]])

    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def add_chars(sentences):
    for sentence_index, sentence in enumerate(sentences):
        for word_index, word_info in enumerate(sentence):
            chars = [c for c in word_info[0]]
            sentences[sentence_index][word_index] = [word_info[0], chars, word_info[1]]
    return sentences


def getCasing(word, lookup):
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(word))

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'

    return lookup[casing]


def create_matrices(sentences, word_index, label_index, case_index, char_index):
    unknown_index = word_index['UNKNOWN_TOKEN']
    padding_index = word_index['PADDING_TOKEN']

    dataset = []

    word_count = 0
    unknownWordCount = 0

    for sentence in sentences:
        word_indices = []
        case_indices = []
        char_indices = []
        label_indices = []

        for word, char, label in sentence:
            word_count += 1
            if word in word_index:
                wordIdx = word_index[word]
            elif word.lower() in word_index:
                wordIdx = word_index[word.lower()]
            else:
                wordIdx = unknown_index
                unknownWordCount += 1
            charIdx = []
            for x in char:
                charIdx.append(char_index[x])
            # Get the label and map to int
            word_indices.append(wordIdx)
            case_indices.append(getCasing(word, case_index))
            char_indices.append(charIdx)
            label_indices.append(label_index[label])

        dataset.append([word_indices, case_indices, char_indices, label_indices])

    return dataset


def padding(sentences):
    max_len = 52
    for sentence in sentences:
        char = sentence[2]
        for x in char:
            max_len = max(max_len, len(x))
    for i, sentence in enumerate(sentences):
        sentences[i][2] = pad_sequences(sentences[i][2], 52, padding='post')
    return sentences
