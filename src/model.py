from keras.models import Model
from keras.layers import Dense, Embedding, Input, Dropout, LSTM, Bidirectional, Lambda
from keras.layers.merge import Concatenate
from src.layers import ChainCRF
import keras.backend as K
from keras.optimizers import Adam,SGD


def get_model(word_embeddings, case_embeddings, char_index, pos_tag_embedding):
    word_ids = Input(batch_shape=(None, None), dtype='int32')
    words = Embedding(input_dim=word_embeddings.shape[0],
                                    output_dim=word_embeddings.shape[1],
                                    mask_zero=True,
                                    weights=[word_embeddings])(word_ids)

    casing_input = Input(batch_shape=(None, None), dtype='int32')
    casing = Embedding(output_dim=case_embeddings.shape[1], input_dim=case_embeddings.shape[0],
                       weights=[case_embeddings], mask_zero=True, trainable=False)(casing_input)

    pos_input = Input(batch_shape=(None, None), dtype='float32')
    pos_tag = Embedding(output_dim=pos_tag_embedding.shape[1], input_dim=pos_tag_embedding.shape[0],
                       weights=[pos_tag_embedding], mask_zero=True, trainable=False)(pos_input)

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

    x = Concatenate(axis=-1)([words, casing, char_embeddings, pos_tag])
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(units=100, return_sequences=True))(x)
    #x = Dense(200, activation='tanh')(x)
    x = Dense(10)(x)

    crf = ChainCRF()
    pred = crf(x)

    model = Model(inputs=[word_ids, casing_input, pos_input, char_input], outputs=[pred])
    model.compile(loss=crf.loss, optimizer=SGD(lr=0.01, clipnorm=5.0))
    model.summary()
    return model
