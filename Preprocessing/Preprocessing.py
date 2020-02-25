'''Extract n-grams of different sizes from a corpus, format them and save relevant ressources to files'''
from nltk.corpus import gutenberg
from nltk.tokenize import RegexpTokenizer
import numpy as np
from pickle import dump


def extract_text(corpus):
    '''Extract text from a file as a string
    :param corpus: input .txt file
    :returns text as string'''

    # Read in text
    text = ''
    with open(corpus) as f:
        text = f.read()

    return text.replace('.', '. <end>')

def extract_text_gutenberg(corpus):
    '''Extract text via nltk
    :param corpus: file id
    :returns text as string'''

    text = gutenberg.raw(corpus).replace('. ', ' . <end> ')
    text = text.replace('? ', ' ? <end> ')
    text = text.replace('! ', ' ! <end> ')
    text = text.replace('?"', ' ? <end> ')
    text = text.replace('!"', ' ! <end> ')
    text = text.replace('."', ' ? <end> ')
    return text


def extract_ngrams(text, n):
    '''Extract n-grams
    :param text: text as string
    :param n: n-gram size
    :returns dictionary with one-hot vectors for input ngrams and correct successive words,
    a list of unique words, number of unique words, word-index mapping'''

    # Dictionary to return
    ngrams_info = dict.fromkeys(['X', 'Y', 'vocab_len', 'tokenizer'])

    tokenizer = RegexpTokenizer(r'\w+')
    # Reserve one token for unknown words
    words = tokenizer.tokenize(text + ' UNK')

    unique_words = np.unique(words)
    unique_word_index = dict((c, i) for i, c in enumerate(unique_words))

    WORD_LENGTH = n
    prev_words = []
    next_words = []

    # feature engineering: number of previous words that determines the next word
    for i in range(len(words) - WORD_LENGTH):
        prev_words.append(words[i:i + WORD_LENGTH])
        next_words.append(words[i + WORD_LENGTH])

    # one hot encoding
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
    '''Extract character sequences
    :param text: text as string
    :param seq_len: length of the character sequences
    :returns dictionary with one-hot encoded character input sequences and correct successive words,
    a list of unique characters, number of unique characters, character-index mapping'''

    # Dictionary to return
    char_info = dict.fromkeys(['X', 'Y', 'unique_characters', 'len_unique_characters', 'unique_characters_index'])

    # Delete 'end-of-sentence' markers
    text = text.replace(' <end> ', ' ')
    text = text.replace('\n', ' ')

    # Split text into character sequences
    character_sequences = []
    unique_characters = list(set(list(text)))

    for i in range(seq_len, len(text)):
        character_sequences.append(text[i - seq_len:i + 1])

    unique_character_index = dict((c, i) for i, c in enumerate(unique_characters))

    SEQUENCE_LENGTH = seq_len
    prev_characters = []
    next_characters = []

    # feature engineering: number of previous characters that determines the next character
    for i in range(len(character_sequences) - SEQUENCE_LENGTH):
        prev_characters.append(character_sequences[i])
        next_characters.append(character_sequences[i + 1][-1])

    # one hot encoding
    X = np.zeros((len(prev_characters), SEQUENCE_LENGTH, len(unique_characters)), dtype=bool)
    Y = np.zeros((len(next_characters), len(unique_characters)), dtype=bool)
    print(X.shape)
    print(Y.shape)
    for i, each_chars in enumerate(prev_characters):
        for j, each_char in enumerate(each_chars):
            X[i, j-1, unique_character_index[each_char]] = 1
        Y[i, unique_character_index[next_characters[i]]] = 1


    char_info['X'] = X
    char_info['Y'] = Y
    char_info['unique_characters'] = unique_characters
    char_info['len_unique_characters'] = len(unique_characters)
    char_info['unique_characters_index'] = unique_character_index

    return char_info


def save_to_file(corpus, sequences, mapping):
    '''Preprocess corpus: save first 1000 lines, 10 character sequences
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
    #save_to_file('chesterton-brown', chesterton_brown['sequences'], chesterton_brown['char_index'])
    #print(load_text('chesterton-brown.txt'))
    print(chesterton_brown)