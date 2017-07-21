from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.callbacks import Callback
from keras.optimizers import adam																																																														
#from keras.utils.data_utils import get_file
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import random
import sys
#from sklearn.cross_validation import train_test_split

filename = "realDonaldTrump.txt"	
text = open(filename).read().lower()        
print('corpus length:', len(text))
chars = sorted(list(set(text)))

# split into input (X) and output (Y) variables
#X = chars
# split into 67% for train and 33% for test
#X_train, X_test = train_test_split(X, test_size=0.20, random_state=42)

print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 140
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
#train, validate, test = np.split(sentences.sample(frac=1), [int(.64*len(sentences)), int(.8*len(sentences))])
print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1	
    y[i, char_indices[next_chars[i]]] = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.36, random_state=42)
X_valid, X_valtest, y_valid, y_valtest = train_test_split(X_test, y_test, test_size = 0.55, random_state=42)
# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(256, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.4))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = adam(lr=0.005) 
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 10):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    filepath="my_model_weights.h5"
    #model.save_weights('')
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='auto')
    callbacks_list = [checkpoint]
    #model.load_weights('my_model_weights.h5')
    model.fit(X_train, y_train, batch_size=128, nb_epoch=20,  callbacks=callbacks_list, validation_data=(X_valid,y_valid),verbose = 1)
    #model.evaluate(X=X_valid, y=y_valid,batch_size=128,nb_epoch=1,show_accuracy=True)
    #print (hist.history)
    start_index = random.randint(0, len(text) - maxlen - 1)


    for diversity in [1.0]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        #length = 10
        #sentence = text[start_index: start_index + maxlen]
        sentence = raw_input(" Enter sentence ")
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
print()
'''model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
#LATER ....

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
'''
