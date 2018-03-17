import numpy as np
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


def get_casing(word, lookup):
    casing = 'other'

    num_of_digits = 0
    for char in word:
        if char.isdigit():
            num_of_digits += 1

    num_of_digits_norm = num_of_digits / float(len(word))

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif num_of_digits_norm > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif num_of_digits > 0:
        casing = 'contains_digit'

    return lookup[casing]


def create_matrices(sentences, word_index, label_index, case_index, char_index):
    unknown_index = word_index['UNKNOWN_TOKEN']
    dataset = []

    word_count = 0
    unknown_word_count = 0

    for sentence in sentences:
        word_indices = []
        case_indices = []
        char_indices = []
        label_indices = []

        for word, char, label in sentence:
            word_count += 1
            if word in word_index:
                word_idx = word_index[word]
            elif word.lower() in word_index:
                word_idx = word_index[word.lower()]
            else:
                word_idx = unknown_index
                unknown_word_count += 1
            char_idx = []
            for x in char:
                char_idx.append(char_index[x])
            # Get the label and map to int
            word_indices.append(word_idx)
            case_indices.append(get_casing(word, case_index))
            char_indices.append(char_idx)
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


def create_batches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches, batch_len


def iterate_mini_batches(dataset, batch_len):
    start = 0
    for i in batch_len:
        tokens = []
        caseing = []
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t, c, ch, l = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            caseing.append(c)
            char.append(ch)
            labels.append(l)
        yield np.asarray(labels), np.asarray(tokens), np.asarray(caseing), np.asarray(char)


def get_words_and_labels(train, val, test):
    label_set = set()
    words = {}

    for dataset in [train, val, test]:
        for sentence in dataset:
            for word, char, label in sentence:
                label_set.add(label)
                words[word.lower()] = True
    return words, label_set
