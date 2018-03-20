from keras.models import Model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D,\
    Flatten, concatenate
from keras.initializers import RandomUniform
from keras.utils import plot_model


def get_model(word_embeddings, case_embeddings, char_index, label_index, posTagEmbedding):
    words_input = Input(shape=(None,), dtype='int32', name='words_input')
    words = Embedding(input_dim=word_embeddings.shape[0], output_dim=word_embeddings.shape[1],
                      weights=[word_embeddings], trainable=False)(words_input)
    casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
    casing = Embedding(output_dim=case_embeddings.shape[1], input_dim=case_embeddings.shape[0],
                       weights=[case_embeddings], trainable=False)(casing_input)
    pos_tag_input = Input(shape=(None,), dtype='int32', name='pos_tag_input')
    pos_tag = Embedding(output_dim=posTagEmbedding.shape[1], input_dim=posTagEmbedding.shape[0],
                       weights=[posTagEmbedding], trainable=False)(pos_tag_input)

    # CNN with character input
    character_input = Input(shape=(None, 52,), name='char_input')
    embed_char_out = TimeDistributed(
        Embedding(len(char_index), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
        name='char_embedding')(character_input)
    dropout = Dropout(0.5)(embed_char_out)
    conv1d_out = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(dropout)
    maxpool_out = TimeDistributed(MaxPooling1D(52))(conv1d_out)
    char = TimeDistributed(Flatten())(maxpool_out)
    char = Dropout(0.5)(char)

    # Bidirectional LSTM
    output = concatenate([words, pos_tag, casing, char])
    output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
    output = TimeDistributed(Dense(len(label_index), activation='softmax'))(output)

    # define the model
    model = Model(inputs=[words_input, pos_tag_input, casing_input, character_input], outputs=[output])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
    model.summary()
    plot_model(model, to_file='model.png')
    return model