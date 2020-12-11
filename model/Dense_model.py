from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import datetime

x = np.load('./npy/all_scale_x_11025sr.npy')
y = np.load('./npy/all_scale_y_11025sr.npy')
x_predict = np.load('./npy/mag_tmp.npy')
y = y - 48
# y = to_categorical(y)


x_predict = x_predict.reshape(1, x_predict.shape[0])

x_train, x_test ,y_train , y_test= train_test_split(x, y, train_size = 0.6)
x_val, x_test ,y_val , y_test= train_test_split(x_test, y_test, train_size = 0.5)

model = Sequential()
model.add(Dense(512,  input_shape=(x_train.shape[1],)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(512 ))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(126))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(37, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], optimizer='adam')
# model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
ealystopping = EarlyStopping(monitor='val_loss',patience=70, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=30,
                             factor=0.5, verbose=1)

strart_time = datetime.datetime.now()
model.fit(x_train, y_train, epochs=10000, batch_size=32, validation_data=(x_val, y_val), callbacks=[ealystopping, reduce_lr])
print('학습 종료시간: ', datetime.datetime.now() - strart_time)
loss, acc=model.evaluate(x_test, y_test, batch_size=32)
model.save('./model/modelLoad/modelFolder/Dense_model1_11025sr.h5')


print("acc",acc)
print("loss",loss)

'''
학습 종료시간:  0:01:44.247005
acc 0.8468112945556641
loss 0.6237873435020447
'''


