from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from pathlib import Path
trn_data_dir = Path('C:/Users/Lalith Chandra A/PycharmProjects/Python_Class_ICPS/Lb2/training')
tst_data_dir = Path('C:/Users/Lalith Chandra A/PycharmProjects/Python_Class_ICPS/Lb2/validation')
trn_genrt = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
tst_genrt = ImageDataGenerator(rescale = 1./255)
trn_set = trn_genrt.flow_from_directory(trn_data_dir,target_size = (64, 64),batch_size = 32,class_mode = 'categorical')
tst_set = tst_genrt.flow_from_directory(tst_data_dir,target_size = (64, 64),batch_size = 32,class_mode = 'categorical')
seq_model = Sequential()
seq_model.add(Conv2D(32, (3, 3),input_shape = (64, 64, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
seq_model.add(Dropout(0.2))
seq_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
seq_model.add(MaxPooling2D(pool_size=(2, 2)))
seq_model.add(Flatten())
seq_model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
seq_model.add(Dropout(0.5))
seq_model.add(Dense(10, activation='softmax'))
epochs = 25
lrate = 0.01
decay_rate = lrate/epochs
sgd_optmzr = SGD(lr=lrate, momentum=0.9, decay=decay_rate, nesterov=False)
seq_model.compile(loss='categorical_crossentropy', optimizer=sgd_optmzr, metrics=['accuracy'])
seq_model_fit = seq_model.fit_generator(trn_set,steps_per_epoch = 1097//32,epochs = 25,validation_data = tst_set,validation_steps = 272//32)