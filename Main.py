"""
Main program to evaluate and text the neural network.
Generate sentences from selected models or get evaluation results
for a selected context / ngram size
"""
import argparse
import pickle

from Generator import Sentence_generator as gen
from Preprocessing import Preprocessing as pre
from keras.models import load_model
import matplotlib.pyplot as plt

# Add arguments
parser = argparse.ArgumentParser(description='A program to run and evaluate the sentence generator')
parser.add_argument('Mode', metavar="program mode",
                    choices=['Evaluation', 'Sentence generation'],
                    type=str, help="Select whether to evaluate the models or whether to generate sentences")
parser.add_argument('N', metavar="n-gram size",
                    choices=[1, 2, 5, 10],
                    type=int, help="Select for which context / ngram-size to run / evaluate the models")

args = parser.parse_args()

# Retrieve selected ngram size
MODE = args.Mode
N = args.N

def load_models(selected_context_size):
    """Loads models for selected context window / word length
    
    :param selected_context_size: selected word length
    :returns both models for the context size"""

    model_austen = None
    model_carroll = None

    if selected_context_size == 1:
        model_austen = load_model('RNN/keras_next_word_model_austen.h1')
        model_carroll = load_model('RNN/keras_next_word_model_carroll.h1')
    elif selected_context_size == 2:
        model_austen = load_model('RNN/keras_next_word_model_austen.h2')
        model_carroll = load_model('RNN/keras_next_word_model_carroll.h2')
    elif selected_context_size == 5:
        model_austen = load_model('RNN/keras_next_word_model_austen.h5')
        model_carroll = load_model('RNN/keras_next_word_model_carroll.h5')
    else:
        model_austen = load_model('RNN/keras_next_char_model_austen.h10')
        model_carroll = load_model('RNN/keras_next_char_model_carroll.h10')

    return(model_austen, model_carroll)

def generate():
    """Generate 5 random sentences for both corpora. Default are the character-based models if
    no ngram size is selected
    """
    if N in [1, 2, 5]:
        text_austen = pre.extract_text_gutenberg('austen-emma.txt')
        ngrams_austen = pre.extract_ngrams(text_austen, N)
        len_unique_words_austen = ngrams_austen['len_unique_words']
        unique_words_austen = ngrams_austen['unique_words']
        unique_word_index_austen = ngrams_austen['unique_word_index']
        sentences_austen = []

        text_carroll = pre.extract_text_gutenberg('carroll-alice.txt')
        ngrams_carroll = pre.extract_ngrams(text_carroll, N)
        len_unique_words_carroll = ngrams_carroll['len_unique_words']
        unique_words_carroll = ngrams_carroll['unique_words']
        unique_word_index_carroll = ngrams_carroll['unique_word_index']
        sentences_carroll = []

        # Generate 5 random sentences of at most 5 words
        for i in range(6):
            sentences_austen.append(gen.gen_random_sent(N, unique_words_austen, unique_word_index_austen, MODEL_AUSTEN, 5))
            sentences_carroll.append(gen.gen_random_sent(N, unique_words_carroll, unique_word_index_carroll, MODEL_CARROLL, 5))

        models = {'austen': MODEL_AUSTEN, 'carroll': MODEL_CARROLL}
        unique_word_indices = [unique_word_index_austen, unique_word_index_carroll]
        austen_best_model = []
        carroll_best_model = []
        austen_best_probs = []
        carroll_best_probs = []

        for s in sentences_austen:
            m, p = gen.assign_author(s, models, unique_word_indices, N)
            austen_best_model.append(m)
            austen_best_probs.append(p)
        for s in sentences_carroll:
            m, p = gen.assign_author(s, models, unique_word_indices, N)
            carroll_best_model.append(m)
            carroll_best_probs.append(p)

        gen.print_sentences(sentences_austen, 'Jane Austen - \'Emma\'',  austen_best_model, austen_best_probs)
        gen.print_sentences(sentences_carroll, 'Lewis Carroll - \'Alice in Wonderland\'', carroll_best_model, carroll_best_probs)
    else:
        text_austen = pre.extract_text_gutenberg('austen-emma.txt')
        ngrams_austen = pre.extract_characters(text_austen, N)
        len_unique_words_austen = ngrams_austen['len_unique_characters']
        unique_words_austen = ngrams_austen['unique_characters']
        unique_word_index_austen = ngrams_austen['unique_characters_index']
        sentences_austen = []

        text_carroll = pre.extract_text_gutenberg('carroll-alice.txt')
        ngrams_carroll = pre.extract_characters(text_carroll, N)
        len_unique_words_carroll = ngrams_carroll['len_unique_characters']
        unique_words_carroll = ngrams_carroll['unique_characters']
        unique_word_index_carroll = ngrams_carroll['unique_characters_index']
        sentences_carroll = []

        # Generate 5 random sentences
        for i in range(6):
            sentences_austen.append(
                gen.gen_random_sent_from_characters(N, unique_words_austen, unique_word_index_austen, MODEL_AUSTEN, 5))
            sentences_carroll.append(
                gen.gen_random_sent_from_characters(N, unique_words_carroll, unique_word_index_carroll, MODEL_CARROLL, 5))

        models = {'austen': MODEL_AUSTEN, 'carroll': MODEL_CARROLL}
        unique_word_indices = [unique_word_index_austen, unique_word_index_carroll]
        austen_best_model = []
        carroll_best_model = []
        austen_best_probs = []
        carroll_best_probs = []

        for s in sentences_austen:
            m, p = gen.assign_author(s, models, unique_word_indices, N)
            austen_best_model.append(m)
            austen_best_probs.append(p)
        for s in sentences_carroll:
            m, p = gen.assign_author(s, models, unique_word_indices, N)
            carroll_best_model.append(m)
            carroll_best_probs.append(p)

        gen.print_sentences(sentences_austen, 'Jane Austen - \'Emma\'', austen_best_model, austen_best_probs)
        gen.print_sentences(sentences_carroll, 'Lewis Carroll - \'Alice in Wonderland\'', carroll_best_model, carroll_best_probs)


def evaluate():
    """Plot accuracy and loss for both models for the selected n-gram size"""

    if N in [1, 2, 5]:
        hist_austen = pickle.load(open("RNN/history_austen.p"+str(N), "rb"))
        hist_carroll = pickle.load(open("RNN/history_carroll.p" + str(N), "rb"))

        plt.plot(hist_austen['acc'])
        plt.title('Training accuracy for ' + str(N) + '-gram model trained on \'Emma\'')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()

        # Plot training & validation loss values
        plt.plot(hist_austen['loss'])
        plt.title('Training loss for ' + str(N) + '-gram model trained on \'Emma\'')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        plt.plot(hist_carroll['acc'])
        plt.title('Training accuracy for ' + str(N) + '-gram model trained on \'Alice in Wonderland\'')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()

        # Plot training & validation loss values
        plt.plot(hist_carroll['loss'])
        plt.title('Training loss for ' + str(N) + '-gram model trained on \'Alice in Wonderland\'')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
    else:
        hist_austen = pickle.load(open("RNN/history_austen_char.p10", "rb"))
        hist_carroll = pickle.load(open("RNN/history_carroll_char.p10" + str(N), "rb"))

        # Plot training accuracy and loss values for 'Emma'
        plt.plot(hist_austen['acc'])
        plt.title('Training accuracy for character model trained on \'Emma\'')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()

        plt.plot(hist_austen['loss'])
        plt.title('Training loss for character model trained on \'Emma\'')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        # Plot training accuracy and loss values for 'Alice in Wonderland'
        plt.plot(hist_carroll['acc'])
        plt.title('Training accuracy for character model trained on \'Alice in Wonderland\'')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()

        plt.plot(hist_carroll['loss'])
        plt.title('Training loss for character model trained on \'Alice in Wonderland\'')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()


# Load models for selected ngram size
MODEL_AUSTEN, MODEL_CARROLL = load_models(N)


if MODE == 'Sentence generation':
    generate()
else:
    evaluate()