# from email import message
from random import sample
from tabnanny import verbose
import pandas as pd # Used for reading the csv data
from nltk.corpus import stopwords
import string # For punctuation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

"""
STEP 1
DATA PREPARATION
"""

df = pd.read_csv("spam.csv", encoding = 'latin-1')

#df.info()

# There are some unwanted extra columns in the data file. To remove them,
df = df.iloc[:, :2]

df.columns = ['target', 'message'] # Change column names


# Sets ham to 0, spam to 1
encoder=LabelEncoder()
df['target']=encoder.fit_transform(df['target'])

# df['message_length'] = df.message.apply(len)

stopwords = stopwords.words("english")

def data_preparation(message):
    """Removes stopwords and punctuations
    Args:
        message (string): message
    Returns:
        string: new cleaned message
    """
    # messages = df["message"] # Messages column
    punctuations = string.punctuation

    words = []
    for word in message.split():
        word = word.lower()
        if word not in stopwords:
            chars = []
            for char in word:
                if char not in punctuations:
                    chars.append(char)
                else:
                    chars.append(" ")
            
            new_word = "".join(chars)
            words.append(new_word) 
    
    new_message = " ".join(words)
    
    return new_message
    

# Add cleaned_messages to df
df['cleaned_message'] = df.message.apply(data_preparation)


"""
STEP 2
MODELLING
"""
targets = df.target
messages = df.cleaned_message
# print(df.cleaned_message[1084])

# Split train and test data
# - train_test_split -
#   - Split arrays or matrices into random train and test subsets
#   - test_size: should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split
#   - random_state: Controls the shuffling applied to the data before applying the split.
#   - stratify: mmm

messages_train, messages_test, targets_train, targets_test = train_test_split(messages, targets, test_size=0.2, random_state=20)

# mx = len(max(messages, key=len))

# Tokenize and padding

num_words = 50000 # The maximum number of words to keep, based on word frequency. 
max_len = 91

tokenizer = Tokenizer(num_words = num_words) 
tokenizer.fit_on_texts(messages_train) # Updates internal vocabulary based on a list of texts.

# Tokenize and paddin for train dataset

messages_train_features = tokenizer.texts_to_sequences(messages_train) # Updates internal vocabulary based on a list of sequences.
# print(len(max(messages_train_features, key=len))) 79
messages_train_features = sequence.pad_sequences(messages_train_features, maxlen = max_len)

# Tokenize and paddin for test dataset

messages_test_features = tokenizer.texts_to_sequences(messages_test)
# print(len(max(messages_test_features, key=len))) #91
messages_test_features = sequence.pad_sequences(messages_test_features, maxlen = max_len)

print(len(messages_train_features), len(messages_train_features[0]))
print(len(messages_test_features), len(messages_test_features[0]))


# Define the Model
model = Sequential()

# Embedding layer
model.add(Embedding(num_words, 32, input_length=max_len))

# LSTM layer
model.add(LSTM(64))

# Dense layer
model.add(Dense(16 ,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))
    
# Add loss function, metrics, optimizer
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=["accuracy"])

print(model.summary())

# messages_train_features.shape,targets_train.shape # shape of train dataset
# messages_test_features.shape,targets_test.shape # shape of test dataset


"""
# TRAINING 
"""
history=model.fit(messages_train_features,targets_train,batch_size=150,epochs=20, validation_data=(messages_test_features, targets_test))

"""
EVALUATION
"""

# Model Performansı

plt.plot(history.history["accuracy"], "--")
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# # Model Hatası

plt.plot(history.history["loss"], "--")
plt.plot(history.history["val_loss"])
plt.title("Model Hatası")
plt.ylabel("Hata")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

accuracy = model.evaluate(messages_test_features,targets_test)
print('Accuracy: {:0.3f}'.format(accuracy[1]*100))

y_predict  = [1 if o>0.5 else 0 for o in model.predict(messages_test_features)]
cf_matrix =confusion_matrix(targets_test,y_predict)
print(cf_matrix)

ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['Not Spam','Spam'])
ax.yaxis.set_ticklabels(['Not Spam','Spam'])

plt.show()



"""
TRY WITH YOUR OWN INPUT
"""
sample_texts = ["Free entry in 2 a weekly competition to win FA Cup final tkts 21st May 2005",
                "i'm not coming to home today",
                "ok. i'm good",
                "congratulations you won! sms YES on 54233 to win",
               "Your 2004 account for 07XXXXXXXXX shows 786",
               "Hey, call ur mom she is worried",
               "YOU ARE CHOSEN TO RECEIVE A å£350 AWARD! Pls call claim to collect your award which you are selected to receive as a valued mobile customer."
               ]
sample_texts = [data_preparation(sentence) for sentence in sample_texts]

txts = tokenizer.texts_to_sequences(sample_texts)
txts = sequence.pad_sequences(txts, maxlen=max_len)
preds = model.predict(txts, verbose=0)
print(np.around(preds))