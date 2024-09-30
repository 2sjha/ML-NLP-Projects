"""
Script to train the chatbot model
"""

import utils as utils
import numpy as np
import tensorflow as tf
from utils import load_qa_dataset


def train_model():
    """
    Trains the model with the custom QnA dataset and saves the model
    """
    qna_df = load_qa_dataset()

    # tokenize text
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(qna_df["question"] + qna_df["answer"])
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    print("Vocab Size : {}".format(VOCAB_SIZE))

    # encoder_input_data
    tokenized_questions = tokenizer.texts_to_sequences(qna_df["question"])
    maxlen_questions = max([len(x) for x in tokenized_questions])
    padded_questions = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_questions, maxlen=maxlen_questions, padding="post"
    )
    encoder_input_data = np.array(padded_questions)

    # decoder_input_data
    tokenized_answers = tokenizer.texts_to_sequences(qna_df["answer"])
    maxlen_answers = max([len(x) for x in tokenized_answers])
    padded_answers = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_answers, maxlen=maxlen_answers, padding="post"
    )
    decoder_input_data = np.array(padded_answers)

    # decoder_output_data
    tokenized_answers = tokenizer.texts_to_sequences(qna_df["answer"])
    for i in range(len(tokenized_answers)):
        tokenized_answers[i] = tokenized_answers[i][1:]
    padded_answers = tf.keras.preprocessing.sequence.pad_sequences(
        tokenized_answers, maxlen=maxlen_answers, padding="post"
    )
    onehot_answers = tf.keras.utils.to_categorical(padded_answers, VOCAB_SIZE)
    decoder_output_data = np.array(onehot_answers)

    encoder_inputs = tf.keras.layers.Input(
        shape=(maxlen_questions,), name="encoder_inputs"
    )
    encoder_embedding_layer = tf.keras.layers.Embedding(
        VOCAB_SIZE, 200, mask_zero=True, name="encoder_embedding"
    )
    encoder_embedding = encoder_embedding_layer(encoder_inputs)
    encoder_lstm = tf.keras.layers.LSTM(200, return_state=True, name="encoder_lstm")
    _, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]

    decoder_inputs = tf.keras.layers.Input(
        shape=(maxlen_answers,), name="decoder_inputs"
    )
    decoder_embedding_layer = tf.keras.layers.Embedding(
        VOCAB_SIZE, 200, mask_zero=True, name="decoder_embedding"
    )
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(
        200, return_state=True, return_sequences=True, name="decoder_lstm"
    )
    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding, initial_state=encoder_states
    )
    decoder_dense = tf.keras.layers.Dense(
        VOCAB_SIZE, activation=tf.keras.activations.softmax, name="decoder_dense"
    )
    output = decoder_dense(decoder_outputs)

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(), loss="categorical_crossentropy"
    )

    model.summary()

    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_output_data,
        batch_size=50,
        epochs=150,
    )
    model.save("model.keras")


if __name__ == "__main__":
    train_model()
