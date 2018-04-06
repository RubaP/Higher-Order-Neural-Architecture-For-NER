from keras.models import Model
from keras.layers import Dense, Embedding, Input, Dropout, LSTM, Bidirectional, Lambda, Add
from keras.layers.merge import Concatenate
from src.model.layers import ChainCRF
import keras.backend as K
from keras.optimizers import SGD
import numpy as np


def get_model(word_embeddings, char_index, pos_tag_index):
    word_ids = Input(batch_shape=(None, None), dtype='int32')
    words = Embedding(input_dim=word_embeddings.shape[0],
                                    output_dim=word_embeddings.shape[1],
                                    mask_zero=True,
                                    weights=[word_embeddings])(word_ids)

    casing_input = Input(batch_shape=(None, None, 11), dtype='float32')

    pos_input = Input(batch_shape=(None, None, len(pos_tag_index)), dtype='float32')

    # build character based word embedding
    char_input = Input(batch_shape=(None, None, None), dtype='int32')
    char_embeddings = Embedding(input_dim=len(char_index),
                                output_dim=25,
                                mask_zero=True
                                )(char_input)
    s = K.shape(char_embeddings)
    char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], 25)))(char_embeddings)

    fwd_state = LSTM(25, return_state=True)(char_embeddings)[-2]
    bwd_state = LSTM(25, return_state=True, go_backwards=True)(char_embeddings)[-2]
    char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])
    # shape = (batch size, max sentence length, char hidden size)
    char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * 25]))(char_embeddings)

    word_representation = Concatenate(axis=-1)([words, char_embeddings, pos_input])
    x = Dropout(0.5)(word_representation)
    x = Bidirectional(LSTM(units=100, return_sequences=True, dropout=0.2, recurrent_dropout=0.5))(x)
    scores = Dense(9)(x)

    crfF = ChainCRF()
    predF = crfF(scores)

    crfB = ChainCRF(go_backwards=True)
    predB = crfB(scores)

    model = Model(inputs=[word_ids, casing_input, pos_input, char_input], outputs=[predF, predB])
    #model.compile(loss=[crfF.loss, crfB.loss], optimizer="adam")
    model.compile(loss=[crfF.loss, crfB.loss], optimizer=SGD(lr=0.01, clipnorm=5.0))
    model.summary()
    return model
