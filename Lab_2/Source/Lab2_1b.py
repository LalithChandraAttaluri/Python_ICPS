import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
admssn_data= pd.read_csv('admissions.csv',header=None).values
x_Train_data, x_Test_data, y_Train_data, y_Test_data = train_test_split(admssn_data[1:,0:8], admssn_data[1:,8],test_size=0.25, random_state=87)
def model_generate():
    seq_model=Sequential()
    seq_model.add(Dense(8,input_dim=8,init='normal',activation='sigmoid'))
    seq_model.add(Dense(13,init='normal',activation='relu'))
    seq_model.add(Dense(1))
    seq_model.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
    return seq_model
tnsr_brd = TensorBoard(log_dir="logslo1/1",histogram_freq=0, write_graph=True, write_images=True)

krs_estmtr=KerasRegressor(build_fn=model_generate)
estmtr=krs_estmtr.fit(x_Train_data,y_Train_data,epochs= 10, batch_size= 130,callbacks=[tnsr_brd])
evltn_score= krs_estmtr.score(x_Test_data,y_Test_data)
print(evltn_score)
plt.plot(estmtr.history['loss'])
plt.title('loss of the model')
plt.ylabel('loss value')
plt.xlabel('epoch')
plt.legend(['Train Data', 'Test Data'], loc='upper left')
plt.show()
