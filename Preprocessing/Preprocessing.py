"""Extract n-grams of different sizes from a corpus and format them"""
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

def extract_text(corpus):
    """Extract text from a file as a string
    :param corpus: input .txt file
    :returns text as string"""

    # Read in text
    text = ''
    with open(corpus) as f:
        text = f.read()
    for w in text.split(' '):
        if w in stopwords:
            text = text.replace(w, ' ')

    return text.replace('.', '. <end>')

def extract_text_gutenberg(corpus):
    """Extract text via nltk
    :param corpus: file id
    :returns text as string"""

    text = gutenberg.raw(corpus).replace('.', ' . <end> ')
    text = text.replace('?', ' ? <end> ')
    text = text.replace('!', ' ! <end> ')

    return text

def extract_words(text, str_len=None):
    """Extract text from a file as a string
    :param text: text as string
    :param str_len (optional): only use substring
    :returns list with indexed input words, correct successive words (one-hot vectors),
    vocabulary length, tokenizer instance"""

    if str_len != None:
        text = text[:str_len]
    # Tokenizer does not filter for 'end-of-sentence' punctuation
    tokenizer = Tokenizer(num_words=1000, filters='@#[\,;:-_~*"]()\t\n', lower=True, oov_token='UNK')
    # Extracts sequences of text
    tokenizer.fit_on_texts([text])
    # Convert sequences of text to sequences of integers
    int_enc = tokenizer.texts_to_sequences([text])[0]

    # Store vocabulary length for embedding layer (+ 1 to encode longest word)
    vocab_len = len(tokenizer.word_index) + 1

    # Create word-word sequences
    sequences = list()
    for i in range(1, len(int_enc)):
        tmp = int_enc[i - 1:i + 1]
        sequences.append(tmp)

    # Longest sequence
    max_len = max([len(s) for s in sequences])
    # Split into first and second element of sequence
    sequences = np.array(sequences)
    X = sequences[:, 0]
    y = sequences[:, 1]

    # Use Keras to_categorical() function to one-hot encode the output / second word
    y = to_categorical(y, num_classes=vocab_len)

    return [X, y, vocab_len, tokenizer, max_len]


def extract_ngrams(text, n, str_len):
    """Extract n-grams
    :param text: text as string
    :param n: n-gram size
    :param str_len (optional): only use substring
    :returns list with indexed input bigrams, correct successive words (one-hot vectors),
    vocabulary length, tokenizer instance, max. sequence length"""

    if str_len != None:
        text = text[:str_len]

    # Tokenizer does not filter for 'end-of-sentence' punctuation
    tokenizer = Tokenizer(num_words=1000, filters='@#[\,;:-_~*"]()\t\n', lower=True, oov_token='UNK')
    # Extracts sequences of text
    tokenizer.fit_on_texts([text])
    # Convert sequences of text to sequences of integers
    int_enc = tokenizer.texts_to_sequences([text])[0]

    # Store vocabulary length for embedding layer (+ 1 to encode longest word)
    vocab_len = len(tokenizer.word_index) + 1

    sequences = list()

    # Create sequences of n words(ngram + successive word)
    for i in range(n, len(int_enc)):
        tmp = int_enc[i - n:i + 1]
        sequences.append(tmp)

    # Find padding length
    max_len = max([len(s) for s in sequences])
    sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

    sequences2 = np.array(sequences)
    X = sequences2[:, :-1]
    y = sequences2[:, -1]

    # Use Keras to_categorical() function to one-hot encode the output / second word
    y = to_categorical(y, num_classes=vocab_len)

    return [X, y, vocab_len, tokenizer, max_len]

def extract_characters(text, str_len):
    """Extract n-grams
    :param text: text as string
    :param str_len (optional): only use substring
    :returns list with indexed input bigrams, correct successive words (one-hot vectors),
    vocabulary length, tokenizer instance, max. sequence length"""

    if str_len != None:
        text = text[:str_len]

    # Tokenizer does not filter for 'end-of-sentence' punctuation
    tokenizer = Tokenizer(num_words=1000, filters='@#[\,;:-_~*"]()\t\n', lower=True, char_level=True, oov_token='UNK')
    # Extracts sequences of text
    tokenizer.fit_on_texts([text])
    # Convert sequences of text to sequences of integers
    int_enc = tokenizer.texts_to_sequences([text])[0]

    # Store vocabulary length for embedding layer (+ 1 to encode longest word)
    vocab_len = len(tokenizer.word_index) + 1

    sequences = list()

    # Create sequences of characters + successive word
    for i in range(1, len(int_enc)):
        tmp = int_enc[i - 1:i + 1]
        sequences.append(tmp)

    # Find padding length
    max_len = max([len(s) for s in sequences])
    sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

    sequences2 = np.array(sequences)
    X = sequences2[:, :-1]
    y = sequences2[:, -1]

    # Use Keras to_categorical() function to one-hot encode the output / second word
    y = to_categorical(y, num_classes=vocab_len)

    return [X, y, vocab_len, tokenizer, max_len]

# Try to generate an entire sentence
def gen_sent(start, max_len, model, tokenizer, limit):
    """Generate sequences of ten tokens starting with the given input text
    :param: start: word to start with
    :param: max_len: length of the longest sequence (for padding)
    :param: model: trained model
    :param: tokenizer: tokenizer initialized with some text
    :param: limit: maximum length of the generated sentence
    :returns generated sentence as a list of tokens"""

    text = start
    sentence = [text]

    # Predict the next word until an 'end-of-sentence' token is predicted or the maximum sequencec length is reached
    for c in range(limit+1):
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
        # Stop the loop when the next word is the 'end-of-sentence' marker
        if next_w == '<end>':
            break

    return sentence