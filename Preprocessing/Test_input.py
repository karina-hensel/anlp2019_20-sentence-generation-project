"""Test format of n-grams for using them in a RNN"""
import pandas
from keras.layers import Dense, LSTM, Embedding
from keras.models import Sequential

import Preprocessing as pre
import Generator.Sentence_generator as gen

def word_model(text, n, num_sent):
    """A model with an embedding, LSTM and linear layer
    :param text: training data
    :param n: n-gram size
    :param num_sent: number of sentences to generate
    :returns generated sentences"""

    words = pre.extract_ngrams(text, n=n)

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
    model.fit(X, y, epochs=5, verbose=2)

    # Generate 5 random sentences
    return gen.gen_random_sent(n, max_len - 1, model, tokenizer, 10, X, num_sent)

def char_model(text, seq_len, num_sent):
    """A model with one LSTM and a linear output layer for character-based generation
    :param text: preprocessed text
    :param seq_len: length of character-sequences in X
    :param num_sent: number of sentences to generate
    :returns generated sentences"""

    chars = pre.extract_characters(text, seq_len)

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
    model.fit(X, y, epochs=10, verbose=2)

    # Generate character sequences
    return gen.gen_random_sent_from_chars(num_chars, seq_len, model, c2i, i2c, 10, num_sent)

def print_sentences(corpus, sentences):
    ''' Print all generated sentences
    :param corpus: text
    :param sentences: generated sentences
    '''
    headers2 = [str(i) for i in range(1, len(sentences)+1)]

    print(corpus + '\n-------------------------')
    print(pandas.DataFrame(sentences, headers2, ['']))
    print()

if __name__ == '__main__':
    # Get text as string
    #text = pre.extract_text_gutenberg('austen-emma.txt')
    text = pre.load_text('../Ressources/chesterton-brown.txt')

    # Test model with characters
    sentences = char_model(text, 10, 5)

    # Test model with single words
    #print(word_model(text, 'she', 1, 10000))
    #sentences = char_model(text, 1, 5)

    # Test model with bigrams
    #print(word_model(text, 'she has', 2, 10000))

    # Test model with trigrams
    #print(word_model(text, 'she has been', 3, 10000))
    print_sentences('chesterton-brown', sentences)