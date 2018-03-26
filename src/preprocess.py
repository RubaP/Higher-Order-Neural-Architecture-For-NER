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
        sentence.append([splits[0], splits[1], splits[-1]])

    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def add_chars(sentences):
    for sentence_index, sentence in enumerate(sentences):
        for word_index, word_info in enumerate(sentence):
            chars = [c for c in word_info[0]]
            sentences[sentence_index][word_index] = [word_info[0], chars, word_info[1], word_info[2]]
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


def create_matrices(sentences, word_index, label_index, case_index, char_index, pos_tag_index):
    unknown_index = word_index['UNKNOWN_TOKEN']
    dataset = []

    word_count = 0
    unknown_word_count = 0

    for sentence in sentences:
        word_indices = []
        case_indices = []
        char_indices = []
        label_indices = []
        pos_tag_inices = []

        for word, char, pos_tag, label in sentence:
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
            pos_tag_inices.append(pos_tag_index[pos_tag])

        dataset.append([word_indices, case_indices, char_indices, label_indices, pos_tag_inices])

    return dataset


def padding(chars):
    padded_chair = []
    for i in chars:
        padded_chair.append(pad_sequences(i, 52, padding='post'))
    return padded_chair


def create_batches(data, batch_size):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    def data_generator():
        """
        Generates a batch iterator for a dataset.
        """
        data_size = len(data)
        while True:
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                X, y = data[start_index: end_index], data[start_index: end_index]
                yield transform(X, y)

    return num_batches_per_epoch, data_generator()


def transform(X, Y):
    max_length_word = max(len(max(seq, key=len)) for seq in Y)

    word_input = []
    char_input = []
    case_input = []
    label_input = []
    pos_tag_input = []

    for word, case, char, label, pos_tag in X:
        word_input.append(pad_sequence(word, max_length_word))
        case_input.append(pad_sequence(case, max_length_word))
        label_input.append(np.eye(10)[pad_sequence(label, max_length_word)])
        pos_tag_input.append(pad_sequence(pos_tag, max_length_word))
        char_input.append(pad_sequence(char, max_length_word, True))

    #print("Word: ", np.asarray(word_input).shape)
    #print("POS tag: ", np.asarray(pos_tag_input).shape)
    #print("Case: ", np.asarray(case_input).shape)
    #print("Char: ", np.asarray(padding(char_input)).shape)
    #print("Label: ", np.asarray(label_input).shape)
    #return [np.asarray(word_input), np.asarray(pos_tag_input), np.asarray(case_input), np.asarray(padding(char_input))], np.asarray(label_input)
    return [np.asarray(word_input), np.asarray(padding(char_input))], np.asarray(label_input)


def pad_sequence(seq, pad_length, isChair = False):
    if isChair:
        for x in range(len(seq), pad_length):
            seq.append([])
        return seq
    else:
        return np.pad(seq, (0, pad_length - len(seq)), 'constant', constant_values=(0,0))


def get_words_and_labels(train, val, test):
    label_set = set()
    pos_tag_set = set()
    words = {}

    for dataset in [train, val, test]:
        for sentence in dataset:
            for word, char, POS_tag, label in sentence:
                label_set.add(label)
                pos_tag_set.add(POS_tag)
                words[word.lower()] = True
    return words, label_set, pos_tag_set
