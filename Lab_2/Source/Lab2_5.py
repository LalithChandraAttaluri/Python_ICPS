import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from sklearn.preprocessing import LabelEncoder
trn_data = pd.read_csv('movie_review_train.csv')
tst_data=pd.read_csv('movie_review_test.csv')
trng_data = trn_data[['phrase','sentiment']]
trng_data['phrase'] = trng_data['phrase'].apply(lambda x: x.lower())
trng_data['phrase'] = trng_data['phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
for idx, row in trng_data.iterrows():
    row[0] = row[0].replace('rt', ' ')
maximum_fatures = 2000
toknzr = Tokenizer(num_words=maximum_fatures, split=' ')
toknzr.fit_on_texts(trng_data['phrase'].values)
T = toknzr.texts_to_sequences(trng_data['phrase'].values)
print(T)
T = pad_sequences(T)
print(T)
embed_dim = 128
lstm_out = 196
def model_generate():
    seq_model = Sequential()
    seq_model.add(Embedding(maximum_fatures, embed_dim,input_length = X.shape[1]))
    seq_model.add(SpatialDropout1D(0.4))
    seq_model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    seq_model.add(Dense(5,activation='softmax'))
    seq_model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    return seq_model
labl_encdr = LabelEncoder()
intgr_encd = labl_encdr.fit_transform(trng_data['sentiment'])
y = to_categorical(intgr_encd)
x_Train_data, x_Test_data, y_Train_data, y_Test_data = train_test_split(T,y, test_size = 0.33, random_state = 42)
print(x_Train_data.shape,y_Train_data.shape)
print(x_Test_data.shape,y_Test_data.shape)

batch_size = 32
sequential_model = model_generate()
history=sequential_model.fit(x_Train_data, y_Train_data, epochs =5, batch_size=batch_size, verbose = 2,validation_data=(X_test,Y_test))
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r--')
plt.legend(['Training Data Loss', 'Validation Data Loss'])
plt.xlabel('Model_Epoch')
plt.ylabel('Model_Loss')
plt.show()