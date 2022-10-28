#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:58:32 2022

@Author:     Alessio Borgi
@Contact :   borgi.1952442@studenti.uniroma1.it
             alessioborgi3@gmail.com
@Filename:   Spam_Detector.py
"""

'''STEP 0: IMPORTING NEEDED LIBRARIES'''
import pandas as pd                                                     #Library used for importing the data using Pandas Dataframes from a csv file.
import re                                                               #
import string                                                           #Library that will be used during the Pre-Processing Step.
import numpy as np                                                      #Library that will be used for manipulation.
from sklearn.model_selection import train_test_split                    #Library that will be used in the Splitting-Step for doing the Train-Test Split.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS          #
from sklearn.preprocessing import LabelEncoder                          #
from keras.models import save_model                                     #Library that will be used for saving the Model once it is trained. 
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional
from keras.models import Model
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix,f1_score, precision_score,recall_score
import seaborn as sns


'''STEP 1: IMPORTING THE DATA'''
data = pd.read_csv('Dataset_SpamHam.csv')
# print(data.head())

data["text"] = data.Email
data["spam"] = data.Label
# print(data.head())

'''STEP 2: SPLITTING THE DATA AND PRE-PROCESSING PIPELINE'''

#SPLITTING DATA:
emails_train, emails_test, target_train, target_test = train_test_split(data.text,data.spam,test_size = 0.2) 
# print(data.info)
# print(emails_train.shape)

#PRE-PROCESSING
def pre_process(word):
    word_without_hyperlink = re.sub(r"http\S+", "", word)
    word_lower = word_without_hyperlink.lower()
    word_without_number = re.sub(r'\d+', '', word_lower)
    word_without_punctuation = word_without_number.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    word_without_spaces = word_without_punctuation.strip()
    f_word = word_without_spaces.replace('\n','')
    
    return f_word


x_train = [pre_process(o) for o in emails_train]
x_test = [pre_process(o) for o in emails_test]
# print(x_train[0])

le = LabelEncoder()
train_y = le.fit_transform(target_train.values)
test_y = le.transform(target_test.values)
# print(train_y)

'''STEP 3: TOKENIZING THE DATA'''

#TOKENIZE:
## some config values 
embed_size = 100 # how big is each word vector
max_feature = 50000 # how many unique words to use (i.e num rows in embedding vector)
max_len = 2000 # max number of words in a question to use

tokenizer = Tokenizer(num_words=max_feature)

tokenizer.fit_on_texts(x_train)

x_train_features = np.array(tokenizer.texts_to_sequences(x_train))
x_test_features = np.array(tokenizer.texts_to_sequences(x_test))
# print(x_train_features[0])

'''STEP 4: PADDING THE DATA'''

#PADDING:
x_train_features = pad_sequences(x_train_features,maxlen=max_len)
x_test_features = pad_sequences(x_test_features,maxlen=max_len)
# print(x_train_features[0])

'''STEP 5: MODEL IMPLEMENTATION'''
#MODEL:
# create the model
embedding_vecor_length = 32

model = tf.keras.Sequential()
model.add(Embedding(max_feature, embedding_vecor_length, input_length=max_len))
model.add(Bidirectional(tf.keras.layers.LSTM(64)))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
history = model.fit(x_train_features, train_y, batch_size=512, epochs=20, validation_data=(x_test_features, test_y))
model.save('Spam_Detector_v_0.0.1')
model_saved = keras.models.load_model('path/to/location')
model_saved('I am so horny!')


'''STEP 6: REVISION: PERFORMANCE FOCUS'''
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()

y_predict  = [1 if o>0.5 else 0 for o in model.predict(x_test_features)]
cf_matrix =confusion_matrix(test_y,y_predict)
tn, fp, fn, tp = confusion_matrix(test_y,y_predict).ravel()

ax= plt.subplot()
sns.heatmap(cf_matrix, annot=True, ax = ax,cmap='Blues',fmt=''); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix')

print("Precision: {:.2f}%".format(100 * precision_score(test_y, y_predict)))
print("Recall: {:.2f}%".format(100 * recall_score(test_y, y_predict)))
print("F1 Score: {:.2f}%".format(100 * f1_score(test_y,y_predict)))
f1_score(test_y,y_predict)
