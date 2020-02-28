"""Several functions to generate sentences with a given model

This script provides several functions for generating sentences from a given
model.

It requires heapq, random, numpy, pandas and nltk to be installed within
the Python environment used to run this project.

The script can be imported as a module and contains the following functions:

    * gen_sent - generate sentences from a specific start ngram
    * gen_random_sent - generate sentences from randomly chosen ngrams
    * gen_random_sent_from_characters - generate sentences from randomly chosen sequence of characters
    * assign_author - find the model, which assigns the highest probability to a sentence
    * print_sentences - print the generated sentences
"""
import heapq
import random

import numpy as np
import pandas
from nltk import RegexpTokenizer


def gen_sent(start, n, unique_words, unique_word_index, model, limit):
    """Generate sequences of tokens starting with the given input text
    (currently not available as an option when running the whole project).

    :param start: ngram to start with
    :type start: str
    :param n: n-gram length
    :type n: int
    :param unique__words: vocabulary
    :type unique_words: list
    :param unique_word_index: word-index mapping
    :type unique_word_index: dict
    :param model: trained model
    :type model: keras.models.Sequential
    :param limit: maximal sentence length
    :type limit: int
    :returns: generated sentence as a list of tokens
    :rtype: list"""

    WORD_LENGTH = n

    # Helper functions to format some input sequence as one-hot vector
    def prepare_input(text):
        x = np.zeros((1, WORD_LENGTH, len(unique_words)))
        for t, word in enumerate(text.split()):
            if word in unique_word_index.keys():
                x[0, t, unique_word_index[word]] = 1
            else:
                x[0, t, unique_word_index['UNK']] = 1
        return x

    # Retrieve best n predictions for the next word from a probability
    # distribution
    def sample(preds, top_n=3):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        return heapq.nlargest(top_n, range(len(preds)), preds.take)

    # Helper function to make predictions for the best next word in
    # the sequence
    def predict_completions(text, n=1):
        if text == "":
            return ("0")
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_indices = sample(preds, n)
        return [unique_words[idx] for idx in next_indices]

    # Initialize tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    text = start
    sentence = [text]

    # Predict the next word until an 'end-of-sentence' token is predicted or the maximum sequence length is reached
    for c in range(limit+1):
        text = " ".join(tokenizer.tokenize(text.lower())[0:WORD_LENGTH])
        next_word = predict_completions(text, 1)
        text = text + " " + next_word[0]
        sentence.append(next_word[0])

    return sentence

def gen_random_sent(n, unique_words, unique_word_index, model, limit):
    """Generate sequences of tokens starting a randomly selected input word

    :param n: n-gram length
    :type n: int
    :param unique__words: vocabulary
    :type unique_words: list
    :param unique_word_index: word-index mapping
    :type unique_word_index: dict
    :param model: trained model
    :type model: keras.models.Sequential
    :param limit: maximal sentence length
    :type limit: int
    :returns: generated sentence as a list of tokens
    :rtype: list"""

    WORD_LENGTH = n

    def prepare_input(text):
        x = np.zeros((1, WORD_LENGTH, len(unique_words)))
        for t, word in enumerate(text.split()):
            if word in unique_word_index.keys():
                x[0, t, unique_word_index[word]] = 1
            else:
                x[0, t, unique_word_index['UNK']] = 1
        return x

    def sample(preds, top_n=3):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        return heapq.nlargest(top_n, range(len(preds)), preds.take)

    def predict_completions(text, n=1):
        if text == "":
            return ("0")
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_indices = sample(preds, n)
        return [unique_words[idx] for idx in next_indices]

    # Initialize tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # Select random words from the vocabulary to build up an ngram of length n
    start = ''
    for i in range(0, n):
        ind = random.randint(0, len(unique_words))
        start += unique_words[ind] + ' '
    text = start
    sentence = [text]

    # Predict the next word until an 'end-of-sentence' token is predicted or the maximum sequence length is reached
    for c in range(limit):
        text = " ".join(tokenizer.tokenize(text.lower())[0:])
        next_word = predict_completions(text, 1)
        text = " ".join(tokenizer.tokenize(text.lower())[1:]) + ' ' + next_word[0]
        sentence.append(next_word[0])

    return sentence

def gen_random_sent_from_characters(n, unique_characters, unique_character_index, model, limit):
    """Generate sequences of characters starting a randomly selected character sequence

    :param n: character sequence length
    :type n: int
    :param unique_characters: vocabulary
    :type unique_characters: list
    :param unique_character_index: character-index mapping
    :type unique_character_index: dict
    :param model: trained model
    :type model: keras.model.Sequential
    :param limit: maximal sequence length
    :type limit: int
    :returns: generated sequence of characters
    :rtype: list"""

    SEQUENCE_LENGTH = n

    def prepare_input(text):
        x = np.zeros((1, SEQUENCE_LENGTH, len(unique_characters)))
        for t, word in enumerate(text.split()):
            if word in unique_character_index.keys():
                x[0, t, unique_character_index[word]] = 1
            else:
                x[0, t, unique_character_index['UNK']] = 1
        return x

    def sample(preds, top_n=3):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        return heapq.nlargest(top_n, range(len(preds)), preds.take)

    def predict_completions(text, n=1):
        if text == "":
            return ("0")
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_indices = sample(preds, n)
        return [unique_characters[idx] for idx in next_indices]

    # Initialize tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # Select n random character from the vocabulary and build up a start character sequence
    start = ''
    for i in range(0, n):
        ind = random.randint(0, len(unique_character_index))
        start += unique_characters[ind]
    text = start
    sentence = [text]

    # Predict the next character until the maximum sequence length is reached
    for c in range(limit):
        text = "".join(tokenizer.tokenize(text.lower())[0:])
        next_word = predict_completions(text, 1)
        text = "".join(tokenizer.tokenize(text.lower())[1:])  + next_word[0]
        sentence.append(next_word[0])

    return sentence

def assign_author(sentence, models, unique_w_ind, word_length):
    """Compute the log-probability of a generated sentence to be written by each author
    represented in the dataset / model which assigns the highest probability to a sentence

    :param sentence: (generated) sentence
    :type sentence: str
    :param models: models (one for each author)
    :type models: dict
    :param unique_w_ind: word-index mappings for each model
    :type unique_w_ind: dict
    :param word_length: n-gram size
    :type word_length: int
    :returns: author of the model which assigns the highest probability to the sentence and log-probability
    :rtype: tuple
    """

    max_prob = float('-inf')
    best_model = ''

    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(" ".join(sentence))
    WORD_LENGTH = word_length

    prev_words = []
    next_words = []

    # Convert sentence to sequences of previous and next words
    for i in range(len(words) - WORD_LENGTH):
        prev_words.append(words[i:i + WORD_LENGTH])
        next_words.append(words[i + WORD_LENGTH])

    # Compute log-probability of the sentence with each model
    for x, (author, model) in enumerate(models.items()):
        log_prob_sent = 0.0
        # Load index mappings for the current model
        unique_word_index = unique_w_ind[x]
        unique_index_word = dict((ind, wrd) for wrd, ind in unique_word_index.items())

        # Convert sentence to one-hot vectors
        X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_word_index)), dtype=bool)
        Y = []

        for next_word in next_words:
            if next_word in unique_word_index.keys():
                Y.append(unique_word_index[next_word])
            else:
                Y.append(unique_word_index['UNK'])

        # Step through the sentence to compute the overall probability
        for i, each_words in enumerate(prev_words):
            for j, each_word in enumerate(each_words):
                if each_word in unique_word_index.keys():
                    X[i, j, unique_word_index[each_word]] = 1
                else:
                    X[i, j, unique_word_index['UNK']] = 1
            p_pred = model.predict(X, verbose=0)[0]
            prob_word = p_pred[Y[i]]
            log_prob_sent += np.log(prob_word)

        if np.exp(log_prob_sent) > max_prob:
            max_prob = np.exp(log_prob_sent)
            best_model = author

    return (best_model, max_prob)

def print_sentences(sentences, author, pred_author, probs):
    """ Print all generated sentences and predicted authors in a
    tabular format

    :param sentences: generated sentences
    :type sentences: list
    :param author: true model
    :type author: str
    :param pred_author: models which assigned the highest probability to each sentence
    :type pred_author: list
    :param probs: highest probability for each sentence
    :type probs: list
    """

    sentences = [' '.join(s) for s in sentences]
    table = {'Sentence': sentences, 'Author': [author]*len(sentences), 'Predicted author': pred_author,
             'Probability': probs}
    df = pandas.DataFrame(data=table)

    print(author + '\n-------------------------')
    print(df.to_string())
