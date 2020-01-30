# Simple LSTM
import sys
print(sys.version)
import keras
print("keras {}".format(keras.__version__))
import tensorflow as tf
print("tensorflow {}".format(tf.__version__))
import numpy as np
print("numpy {}".format(np.__version__))
import matplotlib.pyplot as plt

rom keras.backend.tensorflow_backend import set_session
print(tf.__version__)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.visible_device_list = "0"
#### 1 GPU1
#### 2 GPU2
#### 0 GPU3
#### 4 GPU4
set_session(tf.Session(config=config))

from keras.models import Model
from keras.layers import Input, Dense, Activation, Embedding, LSTM


def define_model(vocab_size,
                 input_length=1,
                 dim_dense_embedding=10,
                 hidden_unit_LSTM=5):
    main_input = Input(shape=(input_length,),dtype='int32',name='main_input')
    embedding = Embedding(vocab_size, dim_dense_embedding,
                         input_length=input_length)(main_input)
    x = LSTM(hidden_unit_LSTM)(embedding)
    main_output = Dense(vocab_size, activation='softmax')(x)
    model = Model(inputs=[main_input],
                  output=[main_output])
    print(model.summary())
    return(model)

model = define_model(vocab_size,
                       input_length=1,
                       dim_dense_embedding=10,
                       hidden_unit_LSTM=10)
# compile network
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# fit network
hist = model.fit(X, y, epochs=500, verbose=False)


plt.plot(hist.history["acc"])
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()


# next: increasing the dimention of the dense embedding vector,
# and increasing the number of hidden units in LSTM.
#dim_dense_embedding=30,
#hidden_unit_LSTM=64
#Check the dimentions of available weights
# dimensionality reduction??
