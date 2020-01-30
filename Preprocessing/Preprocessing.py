'''Neural network to predict the next word in a sequence based on the previous token'''
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, LSTM, ReLU, Embedding, Input


# Read in text
text = gutenberg.raw('austen-emma.txt')[:10000]

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
    tmp = int_enc[i-1:i+1]
    sequences.append(tmp)
print(len(sequences))

# Split into first and second element of sequence
sequences = array(sequences)
first = sequences[:,0]
sec = sequences[:,1]

# Use Keras to_categorical() function to one-hot encode the output / second word
y = to_categorical(sec, num_classes=vocab_len)

# Create the actual model
# Embedding layer to learn the word embeddings from the input; input_length=1 because 1 word at a time is passed to NN
model1 = Sequential()
model1.add(Embedding(vocab_len, 10, input_length=1))
model1.add(LSTM(50))
model1.add(Dense(vocab_len, activation='softmax'))
model1.summary()

# Compile the model: provide loss function and optimizer for training
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model for prediction, i.e. train it (500 epochs here)
model1.fit(first, y, epochs=150, verbose=2)

# Try to generate an entire sentence
def gen_sent(X, y, start):
    start_vec = tokenizer.texts_to_sequences([start.lower()])[0]
    start_vec = array(start_vec)
    sentence = [start]
    pred = model1.predict_classes(start_vec, verbose=0)
    c = 0
    while c < 10:
        next_w = ''
        for ind, wrd in tokenizer.index_word.items():
            if ind == pred:
                sentence.append(wrd)
                next_w = wrd
                print(wrd)
            if c > 10:
                break
        vec = tokenizer.texts_to_sequences([next_w])[0]
        vec = array(vec)
        pred = model1.predict_classes(vec, verbose=0)
        c += 1
    return sentence

print(gen_sent(first, y, 'emma'))
