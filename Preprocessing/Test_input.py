"""Test format of n-grams for using them in a RNN"""
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
import numpy as np
import Preprocessing as pre

def word_model(text, start, n, str_len):
    """A model with an embedding, LSTM and linear layer
    :param text: corpus
    :param start: start sequence
    :param n: n-gram size
    :param str_len: only use part of the corpus if given
    :returns generated sentences"""

    words = pre.extract_ngrams(text, n=n, str_len=str_len)

    X = words['X']
    y = words['Y']
    vocab_len = words['vocab_len']
    tokenizer = words['tokenizer']
    max_len = words['max_len']

    # Create the actual model
    # Embedding layer to learn the word embeddings from the input; input_length needs to be adjusted for n-grams
    model = Sequential()
    model.add(Embedding(vocab_len, 10, input_length=n))
    model.add(LSTM(50))
    model.add(Dense(vocab_len, activation='softmax'))
    model.summary()

    # Compile the model: provide loss function and optimizer for training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model for prediction, i.e. train it (500 epochs here)
    model.fit(X, y, epochs=500, verbose=2)

    return pre.gen_sent(start, max_len - 1, model, tokenizer, limit=10)

def char_model(text, seq_len, str_len):
    """A model with one LSTM and a linear output layer
    :param X: vocabulary of characters
    :param y: successive characters
    :param vocab_len: length of vocabulary
    :returns trained model"""
    chars = pre.extract_characters(text, seq_len, str_len)

    X = chars['X']
    y = chars['Y']
    num_chars = chars['num_chars']
    max_len = chars['max_len']
    seq_len = chars['seq_len']
    c2i = chars['char_index']
    i2c = chars['index_char']
    seq = chars['sequences']

    # Create the actual model
    model = Sequential()
    model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(num_chars, activation='softmax'))

    # Compile the model: provide loss function and optimizer for training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model for prediction, i.e. train it (100 epochs here)
    model.fit(X, y, epochs=100, verbose=2)

    # Generate sentences

    return pre.gen_sent_from_chars('he was the ', num_chars, seq_len, model, c2i, i2c, 10)

if __name__ == '__main__':
    # Get text as string
    text = pre.extract_text_gutenberg('austen-emma.txt')

    # Test model with characters
    print(char_model(text, 10, 10000))

    # Test model with single words
    print(word_model(text, 'she', 1, 10000))

    # Test model with bigrams
    print(word_model(text, 'she has', 2, 10000))

    # Test model with trigrams
    print(word_model(text, 'she has been', 3, 10000))
