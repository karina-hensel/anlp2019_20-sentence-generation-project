"""Extract n-grams of different sizes from a corpus and format them"""
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import gutenberg
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from pickle import dump, load


def extract_text(corpus):
    """Extract text from a file as a string
    :param corpus: input .txt file
    :returns text as string"""

    # Read in text
    text = ''
    with open(corpus) as f:
        text = f.read()

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
    """Extract text from a file as a string (old; use extract_ngrams instead)
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


def extract_ngrams(text, n, str_len=None):
    """Extract n-grams
    :param text: text as string
    :param n: n-gram size
    :param str_len (optional): only use substring
    :returns dictionary with indexed input bigrams, correct successive words (one-hot vectors),
    vocabulary length, tokenizer instance, max. sequence length"""

    # Dictionary to return X, y, , tokenizer, max_len
    ngrams_info = dict.fromkeys(['X', 'Y', 'vocab_len', 'tokenizer', 'max_len'])

    # Only use part of text if specified
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

    ngrams_info['X'] = X
    ngrams_info['Y'] = y
    ngrams_info['vocab_len'] = vocab_len
    ngrams_info['tokenizer'] = tokenizer
    ngrams_info['max_len'] = max_len

    return  ngrams_info

def extract_characters(text, seq_len, str_len):
    """Extract n-grams
    :param text: text as string
    :param seq_len: length of the character sequences
    :param str_len (optional): only use substring
    :returns dictionary with one-hot encoded input sequences, correct successive words (one-hot vectors),
    vocabulary length, sequence length, mappings from characters to index and from index to characters"""

    # Dictionary to return
    char_info = dict.fromkeys(['X', 'Y', 'num_chars', 'max_len', 'seq_len', 'char_index', 'index_char', 'sequences'])

    # Only use part of text if specified
    if str_len != None:
        text = text[:str_len]
    # Delete 'end-of-sentence' markers
    text = text.replace(' <end> ', ' ')

    # Split text into character sequences
    sequ = []
    for i in range(0, len(text)-seq_len):
        sequ.append(text[i:i+seq_len])

    # Length of the longest sentence
    max_len = max(len(text.split(' . ')), max(len(text.split(' ! ')), len(text.split(' ? '))))

    # Extract unique characters
    chars = sorted(list(set(text)))
    num_chars = len(chars)

    # Map characters to indices
    char_index = {c:i for i, c in enumerate(chars)}
    # Reverse
    index_char = {i:c for i, c in enumerate(chars)}

    # Integer encode all sequences
    sequences = list()

    for s in sequ:
        tmp = [char_index[c] for c in s]
        sequences.append(tmp)

    # Create input and output pairs
    sequences = np.array(sequences)
    X = sequences[:, :-1]
    y = sequences[:, -1]

    # One-hot encode each sequence and each successive character
    sequences = [to_categorical(s, num_classes=num_chars) for s in X]
    X = np.array(sequences)
    y = to_categorical(y, num_classes=num_chars)

    char_info['X'] = X
    char_info['Y'] = y
    char_info['num_chars'] = num_chars
    char_info['max_len'] = max_len
    char_info['seq_len'] = seq_len
    char_info['char_index'] = char_index
    char_info['index_char'] = index_char
    char_info['sequences'] = sequ

    return char_info

def gen_sent(start, max_len, model, tokenizer, limit):
    """Generate sequences of tokens starting with the given input text
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


def gen_sent_from_chars(start, num_chars, seq_len, model, char_ind, ind_char, limit):
    """Generate a sentence starting with the given input character sequence
    :param: start: character sequence to start with
    :param: num_chars: unique characters / vocab length
    :param: seq_len: length of characters sequences in X
    :param: model: trained model
    :param char_ind: character-index mapping
    :param ind_char: index-character mapping
    :param: limit: maximum length of the generated sentence
    :returns generated sentence as a string"""

    text = start

    # Predict the next word until an 'end-of-sentence' token is predicted or the maximum sequence length is reached
    for i in range(limit):
        # Integer encode character sequence
        text_enc = [char_ind[c] for c in text]
        # Add padding
        text_enc = pad_sequences([text_enc], maxlen=seq_len-1, truncating='pre')
        #print(len(char_ind))
        text_enc = to_categorical(text_enc, num_classes=num_chars)
        # Predict next word
        pred = model.predict_classes(text_enc, verbose=0)

        # Look up the character of the predicted index and append it to the sequence
        next_c = ''
        for ind, char in ind_char.items():
            if ind == pred:
                next_c = char
                break
        text += next_c

        # Stop the loop when the next word is the 'end-of-sentence' marker
        if next_c == '.' or next_c == '!' or next_c == '?':
            break

    return text

def save_to_file(corpus, sequences, mapping):
    # Write sequences to file
    f = open(corpus + '-seq.txt', 'w')
    for s in sequences:
        f.write(s+'\n')

    # Save character-index mapping as pickle file
    dump(mapping, open(corpus + '-char-to-ind.pkl', 'wb'))

#t = extract_characters(extract_text_gutenberg('austen-emma.txt'), 10, 10000)

#save_to_file('austen-emma', t['sequences'], t['char_index'])