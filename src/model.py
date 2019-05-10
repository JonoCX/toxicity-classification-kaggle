import numpy as np 
import pandas as pd 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

train_data = pd.read_csv('../data/train.csv', usecols = ['id', 'target', 'comment_text'])
test_data = pd.read_csv('../data/test.csv')

# Only consider the comment column for the moment
train_data = train_data[['id', 'target', 'comment_text']]

# set the index to the id
train_data.set_index('id', inplace = True)

# Turn the target into a binary feature
train_y = np.where(train_data['target'] >= 0.5, 1, 0)

# set the test index to id too
test_data.set_index('id', inplace = True)

X_train = train_data['comment_text']
X_test = test_data['comment_text']

max_words = 10000
max_sequence_length = 200

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(X_train)

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)

train_pad = pad_sequences(sequences_train, maxlen=max_sequence_length)
test_pad = pad_sequences(sequences_test, maxlen=max_sequence_length)

print(train_pad)
print(train_pad.shape)
print(test_pad.shape)