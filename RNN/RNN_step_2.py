
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, LSTM, ReLU, Embedding, Input
import matplotlib.pyplot as plt
import numpy as np

def extract_word_vectors(corpus):
    # Read in text
    text = gutenberg.raw(corpus)[:10000]

    # Extract one word and the following one
    tokenizer = Tokenizer()
    # Extracts sequences of text
    tokenizer.fit_on_texts([text])
    # Convert sequences of text to sequences of ints
    int_enc = tokenizer.texts_to_sequences([text])[0]

    # Store vocabulary length for embedding layer (+ 1 to encode longest word)
    vocab_len = len(tokenizer.word_index) + 1

    # Create word-word sequences
    sequences = list()
    for i in range(1, len(int_enc)):
        tmp = int_enc[i - 1:i + 1]
        sequences.append(tmp)

    # Split into first and second element of sequence
    sequences = array(sequences)
    X, y = sequences[:,:-1], sequences[:,-1]

    # Use Keras to_categorical() function to one-hot encode the output / second word
    y = to_categorical(y, num_classes=vocab_len)


    return [X, y, vocab_len, tokenizer]

def model(X, y, vocab_len):
    # Create the actual model
    # Embedding layer to learn the word embeddings from the input; input_length=1 because 1 word at a time is passed to NN
    seq_length = X.shape[1]
    model = Sequential()
    model.add(Embedding(vocab_len, 50, input_length=seq_length))
    model.add(LSTM(800, return_sequences=True))
    model.add(LSTM(800))
    model.add(Dense(y.shape[1], activation='softmax'))
    print(model.summary())

    # Compile the model: provide loss function and optimizer for training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model for prediction, i.e. train it (500 epochs here)
    hist = model.fit(X, y, batch_size=64, epochs=100, verbose=1)

    plt.plot(hist.history["accuracy"])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()

    return model

# Try to generate an entire sentence
def gen_sent(start, model, tokenizer):
    start_vec = tokenizer.texts_to_sequences([start.lower()])[0]
    start_vec = array(start_vec)
    sentence = [start]
    pred = model.predict_classes(start_vec, verbose=0)
    c = 0
    while c < 10:
        next_w = ''
        for ind, wrd in tokenizer.index_word.items():
            if ind == pred:
                sentence.append(wrd)
                next_w = wrd
            if c > 10:
                break
        vec = tokenizer.texts_to_sequences([next_w])[0]
        vec = array(vec)
        pred = model.predict_classes(vec, verbose=0)
        c += 1
    return sentence

# Test
word_vec = extract_word_vectors('austen-emma.txt')
X = word_vec[0]
y = word_vec[1]
vocab_len = word_vec[2]
tokenizer = word_vec[3]
model = model(X, y, vocab_len)


print(gen_sent('emma', model, tokenizer))
