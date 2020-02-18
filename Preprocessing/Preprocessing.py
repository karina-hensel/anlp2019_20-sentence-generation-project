"""Extract n-grams of different sizes from a corpus, format them and save relevant ressources to files"""
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import gutenberg
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from pickle import dump, load
import random


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

    text = gutenberg.raw(corpus).replace('. ', ' . <end> ')
    text = text.replace('? ', ' ? <end> ')
    text = text.replace('! ', ' ! <end> ')
    text = text.replace('?"', ' ? <end> ')
    text = text.replace('!"', ' ! <end> ')
    text = text.replace('."', ' ? <end> ')
    return text

'''def extract_words(text, str_len=None):
    """Extract text from a file as a string (old; use extract_ngrams with n=1 instead)
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

    return [X, y, vocab_len, tokenizer, max_len]'''


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
    tokenizer = Tokenizer(num_words=1000, filters='@#[\,;:-_~*"]()\t\n', lower=True)
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

def extract_characters(text, seq_len, str_len=None):
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
    text = text.replace('\n', ' ')

    # Split text into character sequences
    sequ = []
    for i in range(seq_len, len(text)):
        sequ.append(text[i-seq_len:i+1])

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


def save_to_file(corpus, sequences, mapping):
    '''Preprocess corporus: save first 1000 lines, 10 character sequences
    and character-index mappings to separate files
    :param corpus: piece of text
    :param sequences: character sequences
    :param mapping: character-index dictionary'''

    # Extract first 1000 lines from a text file
    text = extract_text_gutenberg(corpus + '.txt')

    f1 = open('../Ressources/' + corpus + '.txt', 'w')
    c = 0
    for l in text.split('\n'):
        f1.write(l + '\n')
        c += 1
        if c > 1000: break

    # Write sequences to file
    f = open('../Ressources/' + corpus + '-seq.txt', 'w')
    for s in sequences:
        f.write(s+'\n')

    # Save character-index mapping as pickle file
    dump(mapping, open('../Ressources/' + corpus + '-char-to-ind.pkl', 'wb'))

def load_text(file):
    '''Load the contents of a preprocessed corpus
    :param file: text file in Ressources directory
    :returns string representation of file content'''
    f = open('../Ressources/' + file, 'r')

    return " ".join(f.readlines()).replace('\n', '')


if __name__ == '__main__':
    # Preprocess five corpora

    #austen_emma = extract_characters(extract_text_gutenberg('austen-emma.txt'), 10)
    #save_to_file('austen-emma', austen_emma['sequences'], austen_emma['char_index'])

    #bible = extract_characters(extract_text_gutenberg('bible-kjv.txt'), 10)
    #save_to_file('bible-kjv', bible['sequences'], bible['char_index'])

    #shakespeare_hamlet = extract_characters(extract_text_gutenberg('shakespeare-hamlet.txt'), 10)
    #save_to_file('shakespeare-hamlet', shakespeare_hamlet['sequences'], shakespeare_hamlet['char_index'])

    #carroll_alice = extract_characters(extract_text_gutenberg('carroll-alice.txt'), 10)
    #save_to_file('carroll-alice', carroll_alice['sequences'], carroll_alice['char_index'])

    #blake_poems = extract_characters(extract_text_gutenberg('blake-poems.txt'), 10)
    #save_to_file('blake-poems', blake_poems['sequences'], blake_poems['char_index'])

    chesterton_brown = extract_characters(extract_text_gutenberg('chesterton-brown.txt'), 10)
    save_to_file('chesterton-brown', chesterton_brown['sequences'], chesterton_brown['char_index'])
    print(load_text('chesterton-brown.txt'))