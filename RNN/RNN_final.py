import numpy as np
import nltk
from nltk import corpus
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.layers import Dense, Flatten, LSTM, ReLU, Embedding, Input
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq
from keras.layers import Dropout
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.callbacks import LearningRateScheduler
from keras.callbacks import History
from keras.regularizers import l2
from keras import losses
from sklearn.utils import shuffle
from Preprocessing import Preprocessing as pre

#import book and create unique words
text = gutenberg.raw('carroll-alice.txt')[:100000]
print('corpus length:', len(text))

tokenizer = RegexpTokenizer(r'\w+')
words = tokenizer.tokenize(text)

#Feature engineering.
#1. Create unique words & predict accorsing to author style.
unique_words = np.unique(words)
unique_word_index = dict((c, i) for i, c in enumerate(unique_words))



#2. Define word length. Next word depends on the 5 previous words.
WORD_LENGTH = 1
prev_words = []
next_words = []
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])
# print(prev_words[0])
#print(next_words[0])

#One hot encoding
X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
Y = np.zeros((len(next_words), len(unique_words)), dtype=bool)
for i, each_words in enumerate(prev_words):
    for j, each_word in enumerate(each_words):
        X[i, j, unique_word_index[each_word]] = 1
    Y[i, unique_word_index[next_words[i]]] = 1

'''char_info = pre.extract_characters(text, 10)
X =char_info['X']
Y = char_info['Y']
unique_words = char_info['unique_characters']
len_unique_characters = char_info['len_unique_characters']
unique_character_index=char_info['unique_characters_index']'''
#earning_rate = 0.1
#decay_rate = 0.1
#epochs = 2


#RNN Model
#model = Sequential()
#embedding layer
#model.add(LSTM(256, input_shape=(WORD_LENGTH, len(unique_words)))) #many to many
#model.add(Dropout(0.2))
#model.add(Dense(len(unique_words)))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

#Training
optimizer = optimizers.adam(lr=0.1)
model = Sequential()
model.add(LSTM(128, input_shape=(WORD_LENGTH, len(unique_words))))
model.add(Dropout(0.2))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
hist = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=10, shuffle=True).history

model.save('keras_next_word_model_carroll.h1')
pickle.dump(hist, open("history_carroll.p1", "wb"))
# define the learning rate change
#def exp_decay(epoch):
    #lrate = learning_rate * np.exp(-decay_rate*epoch)
    #return lrate

# learning schedule callback
#oss_history = History()
#lr_rate = LearningRateScheduler(exp_decay)
#callbacks_list = [loss_history, lr_rate]


#hist = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=epochs,verbose=1, shuffle=True, callbacks=callbacks_list).history
# Plot training & validation accuracy values
plt.plot(hist['acc'])
plt.plot(hist['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
