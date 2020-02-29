## Generating English Sentences Using LSTMs.

Through the use of Long Short-Term Memory (LSTM) Networks, we aim to predict a word depending on the n previous words of the sentence. 
To achieve this, each corpus is transformed into one-hot encoded sequences of n-grams or characters. After loading a pre-trained model, it 
generates a probability distribution over next words / characters following a n-gram / character sequence. The token / character with the
 highest probability is then appended to the sentence.
 The LSTM has been trained on two books: 'Emma' by Jane Austen and 'Alice in Wonderland' by Lewis Carroll.

## Running the Code

To run the code open the terminal in the project directory where the Main.py script is located.

There are two modes available to run the project.

- <b>Sentence generation mode:</b> 
    - To generate random sentences and see which model assigns the highest probability to each of them, type 
```python3 Main.py 'sentence generation - random' n```
where ```n``` is 1, 2, or 5 for the available ngram models and 10 for the character-based model.
    - To generate a sentence from a given start n-gram , type 
```python3 Main.py 'sentence generation - start' --start <start-sequence>```
where ```n``` is 1, 2, or 5 for the available ngram models (this option is not available for the character-based model) 
and provide a start sequence.

- <b>Evaluation mode:</b> This is the default mode, which generates plots of training accuracy and loss for the models of a selected ngram size. 
To run the program in this mode type ```python3 Main.py 'evaluation' n```, where ```n``` can have one of the values as specified above.

### Dependencies

* `Python 3.6.1`
* `Tensorflow 1.3.0`
* `Keras 2.1.2`
* `matplotlib 2.0.2`
* `numpy 1.12.1`
* `nltk 3.4.5`

## Files Description: 

### Main.py 

Main script to run the program as specified above. It loads both models for a selected input size and allows to evaluate them with regard to the 
generated sentences or accuracy and loss.

### RNN_final.py

Includes RNN network. The network is pretrained for several corpuses and the models are saved and loaded.

RNN arguments:


	|-------------|--------|
	|`Optimizer`|ADAM|
	|`epochs`|number of epochs to train for (default: 10)|
	|`l_r`|Learning rate (default: 0.001)|
	|`activation function`|Softmax|
	|`batch_size`|(default: 128)|
	

### Preprocessing.py

Extract n-grams of different sizes from a corpus and format them, such that they can be used as input
 in an RNN.
 
 ### Sentence_generator.py
 Functions to generate sentences from n-gram or character-based (not optimized in the current version of the project) models.

### Pretrained models (RNN)

The 'RNN' directory contains pretrained models for the two corpora and their evaluation.

Available models:
1. Uni-gram models:
    - keras_next_word_model_austen.h1
    - keras_next_word_model_carroll.h1
2. Bi-gram models:
    - keras_next_word_model_austen.h2
    - keras_next_word_model_carroll.h2
3. 5-gram models:
    - keras_next_word_model_austen.h5
    - keras_next_word_model_carroll.h5
4. Character-based models:
    - keras_next_char_model_austen.h10
    - keras_next_char_model_carroll.h10
