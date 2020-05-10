import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
#Defining size of the batch & size of the epoch
size_btch = 140
nb_cls = 10
nb_epch = 100
hrt_disease_data = pd.read_csv("heart_disesases_details.csv").values
import numpy as np
hrt_disease_data[1:,:]=np.asarray(hrt_disease_data[1:,:],dtype=np.float32)
hrt_disease_data[1:,0]=(hrt_disease_data[1:,0]-min(hrt_disease_data[1:,0]))/(max(hrt_disease_data[1:,0])-min(hrt_disease_data[1:,0]))
hrt_disease_data[1:,1]=(hrt_disease_data[1:,1]-min(hrt_disease_data[1:,1]))/(max(hrt_disease_data[1:,1])-min(hrt_disease_data[1:,1]))
hrt_disease_data[1:,2]=(hrt_disease_data[1:,2]-min(hrt_disease_data[1:,2]))/(max(hrt_disease_data[1:,2])-min(hrt_disease_data[1:,2]))
hrt_disease_data[1:,3]=(hrt_disease_data[1:,3]-min(hrt_disease_data[1:,3]))/(max(hrt_disease_data[1:,3])-min(hrt_disease_data[1:,3]))
hrt_disease_data[1:,4]=(hrt_disease_data[1:,4]-min(hrt_disease_data[1:,4]))/(max(hrt_disease_data[1:,4])-min(hrt_disease_data[1:,4]))
hrt_disease_data[1:,5]=(hrt_disease_data[1:,5]-min(hrt_disease_data[1:,5]))/(max(hrt_disease_data[1:,5])-min(hrt_disease_data[1:,5]))
hrt_disease_data[1:,6]=(hrt_disease_data[1:,6]-min(hrt_disease_data[1:,6]))/(max(hrt_disease_data[1:,6])-min(hrt_disease_data[1:,6]))
hrt_disease_data[1:,7]=(hrt_disease_data[1:,7]-min(hrt_disease_data[1:,7]))/(max(hrt_disease_data[1:,7])-min(hrt_disease_data[1:,7]))
hrt_disease_data[1:,8]=(hrt_disease_data[1:,8]-min(hrt_disease_data[1:,8]))/(max(hrt_disease_data[1:,8])-min(hrt_disease_data[1:,8]))
hrt_disease_data[1:,9]=(hrt_disease_data[1:,9]-min(hrt_disease_data[1:,9]))/(max(hrt_disease_data[1:,9])-min(hrt_disease_data[1:,9]))
hrt_disease_data[1:,10]=(hrt_disease_data[1:,10]-min(hrt_disease_data[1:,10]))/(max(hrt_disease_data[1:,10])-min(hrt_disease_data[1:,10]))
hrt_disease_data[1:,11]=(hrt_disease_data[1:,11]-min(hrt_disease_data[1:,11]))/(max(hrt_disease_data[1:,11])-min(hrt_disease_data[1:,11]))
x_Train_data, x_Test_data, y_Train_data, y_Test_data = train_test_split(hrt_disease_data[:,0:13], hrt_disease_data[:,13],test_size=0.25, random_state=87)
Y_Train = np_utils.to_categorical(y_Train_data, nb_cls)
Y_Test = np_utils.to_categorical(y_Test_data, nb_cls)
seq_model = Sequential()
seq_model.add(Dense(output_dim=10, input_shape=(13,), init='normal', activation='softmax'))
seq_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
seq_model.summary()
tnsr_brd = TensorBoard(log_dir="logslo1/2",histogram_freq=0, write_graph=True, write_images=True)
history=seq_model.fit(x_Train_data, Y_Train, nb_epoch=nb_epch, batch_size=size_btch,callbacks=[tnsr_brd])
evl_score = seq_model.evaluate(x_Test_data, Y_Test, verbose=1)
print('Model Loss is: %.2f, Model Accuracy is: %.2f' % (evl_score[0], evl_score[1]))
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss of the model')
plt.xlabel('epoch')
plt.legend(['Train Data', 'Test Data'], loc='upper left')
plt.show()