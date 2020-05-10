from keras.layers import Dropout
from keras.layers import Flatten
import matplotlib.pyplot as plt
from keras.constraints import maxnorm
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.optimizers import SGD
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re
import pandas as pd
trn_data = pd.read_csv('movie_review_train.csv')
tst_data = pd.read_csv('movie_review_test.csv')
train_data = trn_data[['phrase', 'sentiment']]
train_data['phrase'] = train_data['phrase'].apply(lambda x: x.lower())
train_data['phrase'] = train_data['phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
for idx, row in train_data.iterrows():
    row[0] = row[0].replace('rt', ' ')
max_fatures = 2000
toknzd_data = Tokenizer(num_words=max_fatures, split=' ')
toknzd_data.fit_on_texts(train_data['phrase'].values)
T = toknzd_data.texts_to_sequences(train_data['phrase'].values)
print(T)
T_fnl = pad_sequences(T)
print(T_fnl)
embed_dim = 128
lstm_out = 196
print(T_fnl.shape)
def model_generate():
    seq_model = Sequential()
    seq_model.add(Embedding(max_fatures, embed_dim, input_length=T_fnl.shape[1]))
    seq_model.add(
        Conv1D(128, (5), activation='relu', kernel_constraint=maxnorm(3)))
    seq_model.add(Dropout(0.2))
    seq_model.add(Conv1D(128, (5), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
    seq_model.add(MaxPooling1D(5))
    seq_model.add(Flatten())
    seq_model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    seq_model.add(Dropout(0.5))
    seq_model.add(Dense(5, activation='softmax'))
    return seq_model
label__encoder = LabelEncoder()
integer_encoded = label__encoder.fit_transform(train_data['sentiment'])
y = to_categorical(integer_encoded)
x_Train_data, x_Test_data, y_Train_data, y_Test_data = train_test_split(T_fnl, y, test_size=0.33, random_state=42)
print(x_Train_data.shape, y_Train_data.shape)
print(x_Test_data.shape, y_Test_data.shape)
epochs = 15
lrate = 0.01
decay = lrate / epochs
sequntl_model = model_generate()
s_g_d = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
sequntl_model.compile(loss='categorical_crossentropy', optimizer=s_g_d, metrics=['accuracy'])
btch_size = 32
history = sequntl_model.fit(x_Train_data, y_Train_data, epochs=5, batch_size=btch_size, verbose=2)
prfmnce, accrcy = sequntl_model.evaluate(x_Test_data, y_Test_data, verbose=2, batch_size=btch_size)
print(prfmnce)
print(accrcy)
epoch__count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch__count, history.history['loss'], 'r--')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Model_Epoch')
plt.ylabel('Model__Loss')
plt.show()