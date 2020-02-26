import heapq
import random

import numpy as np
import pandas
from nltk import RegexpTokenizer


def gen_sent(start, n, unique_words, unique_word_index, model, limit):
    """Generate sequences of tokens starting with the given input text
    :param: start: word to start with
    :param: n: n-gram length
    :param: unique__words: vocabulary
    :param: unique_word_index: word-index mapping
    :param: model: trained model
    :param: limit: maximal sentence length
    :returns generated sentence as a list of tokens"""

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
    :param: n: n-gram length
    :param: unique__words: vocabulary
    :param: unique_word_index: word-index mapping
    :param: model: trained model
    :param: limit: maximal sentence length
    :returns generated sentence as a list of tokens"""

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

    # Select a random word from the vocabulary
    ind = random.randint(0, len(unique_words))
    start = unique_words[ind] + ' '
    text = start
    sentence = [text]

    # Predict the next word until an 'end-of-sentence' token is predicted or the maximum sequence length is reached
    for c in range(limit+1):
        text = " ".join(tokenizer.tokenize(text.lower())[0:WORD_LENGTH])
        next_word = predict_completions(text, 1)
        text = text + " " + next_word[0]
        sentence.append(next_word[0])

    return sentence

def assign_author(sentence, models, unique_w_ind, word_length):
    '''Compute the probability of a generated sentence to be written by each author
    represented in the dataset / model which assigns the highest probability to a sentence
    :param sentence: (generated) sentence
    :param models: list of models (one for each author)
    :param unique_w_ind: list of word-index mappings for each model
    :param word_length: n-gram size
    :returns index of the model which assigns the highest probability to the sentence'''

    max_prob = float('-inf')
    best_model = ''

    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(sentence)
    WORD_LENGTH = word_length

    prev_words = []
    next_words = []

    # Convert sentence to sequences of previous and next words
    for i in range(len(words) - WORD_LENGTH):
        prev_words.append(words[i:i + WORD_LENGTH])
        next_words.append(words[i + WORD_LENGTH])

    # Compute probability of the sentence with each model
    for x, author, model in enumerate(models.items()):
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

def print_sentences(corpus, sentences, pred_author, probs):
    ''' Print all generated sentences
    :param corpus: text
    :param sentences: generated sentences
    :param pred_author: list of models which assigned the highest probability to each sentence
    :param probs: highest probability for each sentence
    '''
    table = {'Sentence': sentences, 'Author': [corpus]*len(sentences), 'Predicted author': pred_author,
             'Probability': probs}
    df = pandas.DataFrame(data=table)
    headers2 = [str(i) for i in range(1, len(sentences)+1)]

    print(corpus + '\n-------------------------')
    print(pandas.DataFrame(sentences, headers2, ['']))
    #print(df.to_string())
    #df.to_csv(corpus + '.tsv', sep='\t')
