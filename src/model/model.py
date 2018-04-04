from keras.models import Model
from keras.layers import Dense, Embedding, Input, Dropout, LSTM, Bidirectional, Lambda
from keras.layers.merge import Concatenate
from src.model.layers import ChainCRF
import keras.backend as K
from keras.optimizers import SGD


def get_model(word_embeddings, char_index, pos_tag_index):
    word_ids = Input(batch_shape=(None, None), dtype='int32')
    words = Embedding(input_dim=word_embeddings.shape[0],
                                    output_dim=word_embeddings.shape[1],
                                    mask_zero=True,
                                    weights=[word_embeddings])(word_ids)

    casing_input = Input(batch_shape=(None, None, 14), dtype='float32')

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

    x = Concatenate(axis=-1)([words, char_embeddings])
    x = Dropout(0.5)(x)
    x = Concatenate(axis=-1)([x, casing_input, pos_input])
    x = Bidirectional(LSTM(units=100, return_sequences=True))(x)
    x = Dense(9)(x)

    crf = ChainCRF()
    pred = crf(x)

    model = Model(inputs=[word_ids, casing_input, pos_input, char_input], outputs=[pred])
    #model.compile(loss=crf.loss, optimizer=SGD(lr=0.01, clipnorm=5.0))
    model.compile(loss=crf.loss, optimizer="adam")
    model.summary()
    return model
