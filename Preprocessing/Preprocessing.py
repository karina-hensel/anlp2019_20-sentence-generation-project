"""
Data preprocessing functions.

The script provides functions to load and preprocess Project Gutenberg text corpora from
the nltk package and to extract ngrams and character sequences from raw text, such that they
can be used as input to a neural network.

It requires nltk, numpy and pickle to be installed within the Python environment used to run
this project.

The script can be imported as a module and contains the following functions:

    * extract_text - extract text from any text file
    * extract_text_gutenberg - retrieve and preprocess Project Gutenberg corpora
    * extract_ngrams - extract ngrams from string
    * extract_characters - extract character sequences from string
    * save_to_file - save preprocessed Project Gutenberg text
    * load_text - load input from a text file
"""

from nltk.corpus import gutenberg
from nltk.tokenize import RegexpTokenizer
import numpy as np
from pickle import dump


def extract_text(corpus):
    """Extract text from a file as a string
    (currently not used)

    :param corpus: path to input file
    :type corpus: str
    :returns: text as string
    :rtype: str"""

    # Read in text
    text = ''
    with open(corpus) as f:
        text = f.read()

    return text.replace('.', '. <end>')

def extract_text_gutenberg(corpus):
    """Extract text via nltk (only retrieves part of the corpus)

    :param corpus: file id as given in nltk documentation
    :type corpus: str
    :returns: preprocessed text
    :rtype: str
    """

    text = gutenberg.raw(corpus)[:100000].replace('. ', ' . <end> ')
    text = text.replace('? ', ' ? <end> ')
    text = text.replace('! ', ' ! <end> ')
    text = text.replace('?"', ' ? <end> ')
    text = text.replace('!"', ' ! <end> ')
    text = text.replace('."', ' ? <end> ')
    return text


def extract_ngrams(text, n):
    """Extract n-grams and successive words to be used for training the RNN and for
    prediction

    :param text: text
    :type text: str
    :param n: n-gram size
    :type n: int
    :returns: one-hot vectors for input ngrams and correct successive words,
    a list of unique words, number of unique words, word-index mapping
    :rtype: dict
    """

    # Dictionary to return
    ngrams_info = dict.fromkeys(['X', 'Y', 'vocab_len', 'tokenizer'])

    # Tokenize input
    tokenizer = RegexpTokenizer(r'\w+')
    # Reserve one token for unknown words
    words = tokenizer.tokenize(text + ' UNK')

    unique_words = np.unique(words)
    unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

    WORD_LENGTH = n
    prev_words = []
    next_words = []

    # Split the text into sequences of size WORD_LENGTH and the correct successive word
    for i in range(len(words) - WORD_LENGTH):
        prev_words.append(words[i:i + WORD_LENGTH])
        next_words.append(words[i + WORD_LENGTH])

    # One hot encoding of ngram sequences and successive words
    X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
    Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)

    for i, each_words in enumerate(prev_words):
        for j, each_word in enumerate(each_words):
            X[i, j, unique_word_index[each_word]] = 1
        Y[i, unique_word_index[next_words[i]]] = 1

    ngrams_info['X'] = X
    ngrams_info['Y'] = Y
    ngrams_info['unique_words'] = unique_words
    ngrams_info['len_unique_words'] = len(unique_words)
    ngrams_info['unique_word_index'] = unique_word_index

    return  ngrams_info

def extract_characters(text, seq_len):
    """Extract character sequences and successive characters used
    for training and prediction in the character-based RNN.
    Still needs further improvement to produce better results.

    :param text: text as string
    :type text: str
    :param seq_len: length of the character sequences which is used to predict the following character
    :type seq_len: int
    :returns one-hot encoded character input sequences and correct successive character,
    a list of unique characters, number of unique characters, character-index mapping
    :rtype: dict
    """

    # Dictionary to return
    char_info = dict.fromkeys(['X', 'Y', 'unique_characters', 'len_unique_characters', 'unique_characters_index'])

    # Delete 'end-of-sentence' markers
    text = text.replace(' <end> ', ' ')
    text = text.replace('\n', ' ')

    # Split text into character sequences and successive characters
    SEQUENCE_LENGTH = seq_len
    prev_characters = []
    unique_characters = list(set(list(text)))
    unique_characters.append('UNK')
    next_characters = []


    for i in range(seq_len, len(text)-1):
        prev_characters.append([text[i - seq_len:i + 1]])
        next_characters.append(text[i + 1])

    unique_character_index = dict((c, i) for i, c in enumerate(unique_characters))

    # One hot encoding of character sequences and successive character
    X = np.zeros((len(prev_characters), SEQUENCE_LENGTH, len(unique_characters)), dtype=bool)
    Y = np.zeros((len(next_characters), len(unique_characters)), dtype=bool)

    for i, each_chars in enumerate(prev_characters):
        for j, each_char in enumerate(each_chars):
            X[i, j-1, unique_character_index[each_char[j]]] = 1
        Y[i, unique_character_index[next_characters[i]]] = 1


    char_info['X'] = X
    char_info['Y'] = Y
    char_info['unique_characters'] = unique_characters
    char_info['len_unique_characters'] = len(unique_characters)
    char_info['unique_characters_index'] = unique_character_index

    return char_info


def save_to_file(corpus, sequences, mapping):
    """Save first 1000 lines, 10-character sequences
    and character-index mappings to separate files
    (deprecated)

    :param corpus: piece of text
    :type corpus: str
    :param sequences: character sequences
    :type sequences: list
    :param mapping: character-to-index mapping
    :type mapping: dict
    """

    # Extract first 1000 lines from a text file
    text = extract_text_gutenberg(corpus + '.txt')

    f1 = open('../Ressources/' + corpus + '.txt', 'w')
    c = 0
    for l in text.split('\n'):
        print(l)
        f1.write(l + '\n')
        c += 1
        if c > 1000:
            break

    # Write sequences to file
    f = open('../Ressources/' + corpus + '-seq.txt', 'w')
    for s in sequences:
        f.write(s+'\n')

    # Save character-index mapping as pickle file
    dump(mapping, open('../Ressources/' + corpus + '-char-to-ind.pkl', 'wb'))

def load_text(file):
    """Load the contents of a preprocessed corpus
    (deprecated)

    :param file: path to text file in Ressources directory
    :type file: str
    :returns: file content as one line string
    :rtype: str"""
    f = open('../Ressources/' + file, 'r')

    return " ".join(f.readlines()).replace('\n', '')