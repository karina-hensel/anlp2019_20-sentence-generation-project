"""Test format of n-grams for using them in a RNN"""
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
import numpy as np
import Preprocessing as pre

def model(X, y, vocab_len, input_len):
    """A model with an embedding, LSTM and linear layer
    :param X: vocabulary of words / ngrams /characters
    :param y: successive words
    :param vocab_len: length of vocabulary
    :param input_len: n-gram size (1 for single words)
    :returns trained model"""

    # Create the actual model
    # Embedding layer to learn the word embeddings from the input; input_length needs to be adjusted for n-grams
    model = Sequential()
    model.add(Embedding(vocab_len, 10, input_length=input_len-1))
    model.add(LSTM(50))
    model.add(Dense(vocab_len, activation='softmax'))
    model.summary()

    # Compile the model: provide loss function and optimizer for training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model for prediction, i.e. train it (500 epochs here)
    model.fit(X, y, epochs=300, verbose=2)

    return model

if __name__ == '__main__':
    # Get text as string
    text = pre.extract_text_gutenberg('austen-emma.txt')

    # Test model with characters
    #words = pre.extract_characters(text, 10000)
    # Test model with single words
    #words = pre.extract_words(text, 10000)
    # Test model with bigrams
    #words = pre.extract_ngrams(text, n=2, str_len=10000)
    # Test model with trigrams
    words = pre.extract_ngrams(text, n=3, str_len=10000)

    X = words[0]
    y = words[1]
    vocab_len = words[2]
    tokenizer = words[3]
    max_len = words[4]
    #print(X)
    model = model(X, y, vocab_len, max_len)

    #print(pre.gen_sent('she', max_len - 1, model, tokenizer, limit=10))
    print(pre.gen_sent('she has been', max_len - 1, model, tokenizer, limit=10))
    #print(pre.gen_sent('s', max_len - 1, model, tokenizer, limit=10))