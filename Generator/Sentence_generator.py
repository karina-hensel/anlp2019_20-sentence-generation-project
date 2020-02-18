import random

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


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

    # Predict the next word until an 'end-of-sentence' token is predicted or the maximum sequence length is reached
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

def gen_random_sent(n, max_len, model, tokenizer, limit, X, num_sent):
    """Generate 5 sentences, each starting with randomly selected start token(s)
    :param: n-gram size
    :param: max_len: length of the longest sequence (for padding)
    :param: model: trained model
    :param: tokenizer: tokenizer initialized with some text
    :param: limit: maximum length of the generated sentence
    :param: X: observed n-grams
    :param: num_sent: number of sentences to generate
    :returns generated sentence as a list of tokens"""

    sentences = []
    for i in range(0, num_sent):
        # Randomly select an (observed) n-gram
        text = ''
        ind = random.randint(0, len(X))
        start = ''
        for j in range(0, n):
            start += tokenizer.index_word[X[ind][j]] + ' '
        text += start[:-1]
        sentence = [text]

        # Predict the next word until an 'end-of-sentence' token is predicted or the maximum sequence length is reached
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
        sentences.append(text)

    return sentences

def gen_random_sent_from_chars(num_chars, seq_len, model, char_ind, ind_char, limit, num_sent):
    """Generate a sentence starting with the given input character sequence
    :param: num_chars: unique characters / vocab length
    :param: seq_len: length of characters sequences in X
    :param: model: trained model
    :param char_ind: character-index mapping
    :param ind_char: index-character mapping
    :param: limit: maximum length of the generated sentence
    :param: num_sent: number of sentences to generate
    :returns generated sentence as a string"""

    sentences = []

    for i in range(0, num_sent):
        ind = random.randint(0, num_chars-1)
        start = ind_char[ind]
        text = start

        # Predict the next word until an 'end-of-sentence' token is predicted or the maximum sequence length is reached
        for i in range(0, limit):
            # Integer encode character sequence
            text_enc = [char_ind[c] for c in text]
            # Add padding
            text_enc = pad_sequences([text_enc], maxlen=seq_len, truncating='pre')
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

        sentences.append(text)

    return sentences