from keras.models import Model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D,\
    Flatten
from keras.initializers import RandomUniform
from keras.utils import plot_model
from keras.layers.merge import Concatenate
from src.layers import ChainCRF


def get_model(word_embeddings, case_embeddings, char_index, posTagEmbedding, batch_size):
    word_ids = Input(batch_shape=(None, None), dtype='int32')
    words = Embedding(input_dim=word_embeddings.shape[0],
                                    output_dim=word_embeddings.shape[1],
                                    mask_zero=True,
                                    weights=[word_embeddings])(word_ids)

    casing_input = Input(batch_shape=(None, None), dtype='int32')
    casing = Embedding(output_dim=case_embeddings.shape[1], input_dim=case_embeddings.shape[0],
                       weights=[case_embeddings], mask_zero=True,)(casing_input)

    x = Concatenate(axis=-1)([words, casing])
    x = Bidirectional(LSTM(units=200, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = Dense(200, activation='tanh')(x)
    x = Dense(10)(x)

    crf = ChainCRF()
    pred = crf(x)

    model = Model(inputs=[word_ids, casing_input], outputs=[pred])
    model.compile(loss=crf.loss, optimizer="nadam")
    model.summary()
    return model