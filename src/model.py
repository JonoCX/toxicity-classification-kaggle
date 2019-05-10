import numpy as np 
import pandas as pd 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Embedding, Dense, Flattern, BatchNormalization

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

model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))

model.add(Flattern())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit(train_pad, train_y, epochs=10, batch_size=1024, validation_split=0.25, verbose=2)

test_pred = model.predict(pad_test)

sample_result = pd.DataFrame()
sample_result['id'] = test_data.index 
sample_result['prediction'] = test_pred

sample_result.to_csv('../data/submission-1.csv', index=False)