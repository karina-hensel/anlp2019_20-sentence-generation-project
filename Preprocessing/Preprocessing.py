'''Neural network to predict the next word in a sequence based on the previous token'''
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, LSTM, ReLU, Embedding, Input

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
    X = sequences[:, 0]
    y = sequences[:, 1]

    # Use Keras to_categorical() function to one-hot encode the output / second word
    y = to_categorical(y, num_classes=vocab_len)

    return [X, y, vocab_len, tokenizer]

def extract_word_vectors2(corpus, num_sent):
    '''Use bigrams as input (not completely working at the moment)'''
    sents = gutenberg.sents(corpus)[:num_sent]
    text = ''

    for s in sents:
        for w in s:
            text += ' ' + w

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
    X = sequences[:, 0]
    y = sequences[:, 1]

    # Use Keras to_categorical() function to one-hot encode the output / second word
    y = to_categorical(y, num_classes=vocab_len)

    return [X, y, vocab_len, tokenizer]


def extract_trigram_vectors(corpus, num_sent):
    '''Read in the specified number of sentences from the corpus'''
    sents = gutenberg.sents(corpus)[:num_sent]
    #text = gutenberg.raw(corpus)[:num_sent]
    text = ''
    for s in sents:
        for w in s:
            text += ' ' + w
    # Extract one word and the following one
    tokenizer = Tokenizer()
    # Extracts sequences of text
    tokenizer.fit_on_texts([text])
    # Convert sequences of text to sequences of ints
    int_enc = tokenizer.texts_to_sequences([text])[0]

    # Store vocabulary length for embedding layer (+ 1 to encode longest word)
    vocab_len = len(tokenizer.word_index) + 1

    sequences = list()

    # Create 3 word sequences
    for i in range(2, len(int_enc)):
        tmp = int_enc[i - 2:i + 1]
        sequences.append(tmp)

    # Find padding length
    max_len = max([len(s) for s in sequences])
    sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

    sequences2 = array(sequences)
    X = sequences2[:, :-1]
    y = sequences2[:, -1]

    # Use Keras to_categorical() function to one-hot encode the output / second word
    y = to_categorical(y, num_classes=vocab_len)

    return [X, y, vocab_len, tokenizer, max_len]

def model(X, y, vocab_len, input_len):
    # Create the actual model
    # Embedding layer to learn the word embeddings from the input; input_length=1 because 1 word at a time is passed to NN
    model = Sequential()
    model.add(Embedding(vocab_len, 10, input_length=input_len-1))
    model.add(LSTM(50))
    model.add(Dense(vocab_len, activation='softmax'))
    model.summary()

    # Compile the model: provide loss function and optimizer for training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model for prediction, i.e. train it (500 epochs here)
    model.fit(X, y, epochs=100, verbose=2)

    return model

# Try to generate an entire sentence
def gen_sent(start, max_len, model, tokenizer):
    '''Generate sequences of ten tokens starting with the given input text'''
    text = start
    sentence = [text]
    # Predict the next word until a sequence of input_len + 10 words is generated
    for c in  range(11):
        # Retrieve vector of current word
        vec = tokenizer.texts_to_sequences([text])[0]
        # Add padding
        vec = pad_sequences([vec], maxlen=max_len, padding='pre')
        # Predict next word
        pred = model.predict_classes(vec, verbose=0)

        # Look up the word of the predicted index and append it to the sequence
        next_w = ''
        for ind, wrd in tokenizer.index_word.items():
            if ind == pred:
                sentence.append(wrd)
                next_w = wrd
                break
        text += ' ' + next_w

    return sentence

# Test
#word_vec = extract_word_vectors2('austen-emma.txt', 500)
word_vec = extract_trigram_vectors('austen-emma.txt', 500)

X = word_vec[0]
y = word_vec[1]
vocab_len = word_vec[2]
tokenizer = word_vec[3]
max_len = word_vec[4]
model = model(X, y, vocab_len, max_len)

print(gen_sent('emma has', max_len-1, model, tokenizer))