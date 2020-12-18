from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, LSTM, Flatten, MaxPooling2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
from tensorflow.keras.activations import relu,selu, elu

from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad


x = np.load('./npy/all_scale_x_final.npy')
y = np.load('./npy/all_scale_y_final.npy')
x_predict = np.load('./npy/mag_tmp.npy')
y = y - 48
# y = to_categorical(y)


x_predict = x_predict.reshape(1, x_predict.shape[0])

x_train, x_test ,y_train , y_test= train_test_split(x, y, train_size = 0.6)
x_val, x_test ,y_val , y_test= train_test_split(x_test, y_test, train_size = 0.5)

#하이퍼 파라미터 튜닝을 할 모델 작성
def build_model(optimizer = Adam,  epochs = 100,
                loss = 'sparse_categorical_crossentropy',
                patience = 5, layer_num = 2, nodes = 32, activation = Activation('relu'), 
                kernel_init = 'he_normal', k_reg = regularizers.l1, k_reg_param = 0.001,
                lr = 0.001, dropout = 0.1
                 ):
    model = Sequential()
    model.add(Dense(128,  input_shape=(x_train.shape[1],)))
    model.add(BatchNormalization())
    if type(activation) == type('str'):
            model.add(Activation(activation))
    else:
        model.add(activation(alpha=0.3))


    for i in range(layer_num):
        if i == layer_num-1:
            model.add(Dense(nodes, kernel_regularizer=k_reg(k_reg_param) ))
        else:
            if type(kernel_init) == type('str'):
                model.add(Dense(nodes, kernel_initializer=(kernel_init) ))
            else:
                model.add(Dense(nodes, kernel_initializer=kernel_init(seed=None) ))

        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        if type(activation) == type('str'):
            model.add(Activation(activation))
        else:
            model.add(activation(alpha=0.3))

        

    model.add(Dense(37, activation='softmax'))

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy', metrics=['acc'], 
    optimizer=optimizer(learning_rate = lr))
    # model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')

    return model

def create_hyperparameter():
    optimizer = [Adam, RMSprop, Adagrad]
    batchs = [16,32,64,128,216,512]
    dropout = [0.2,0.3,0.4,0.5]
    epochs = [i for i in range(100, 2000, 100)]
    loss = ['sparse_categorical_crossentropy']
    patience = [i for i in range(30, 100, 5)]
    layer_num = [i for i in range(2,5)]
    nodes = [16, 32, 64, 128, 216, 512]
    activation = [
                    'relu',
                    'selu', 
                    'elu', 
                    tf.keras.layers.LeakyReLU,
                    ]
    kernel_init = ['he_normal',
                    'he_uniform',
                    'random_uniform',
                    tf.keras.initializers.GlorotNormal, #xavier nomal
                    tf.keras.initializers.GlorotUniform #xavier uniform
                    ]
    k_reg = [
          regularizers.l1, 
          regularizers.l2
          ]
    k_reg_param = [
        0.001,
        0.003,
        0.005
    ]
    lr = [0.01, 0.008,
                    0.006, 0.005, 0.004,
                    0.003, 0.002, 0.001,
                    0.03, 0.05, 0.06
                    ]
    return      { 
        'batch_size': batchs, 
        'optimizer' : optimizer,  'epochs' : epochs, 'loss' : loss,
        'patience' : patience, 'layer_num' : layer_num, 'nodes' : nodes, 
        'activation' : activation,
        'kernel_init' : kernel_init, 
        'k_reg' : k_reg , 
        'k_reg_param' : k_reg_param,
        'lr': lr,
        'dropout': dropout
    }


hyperparameter = create_hyperparameter()
model = KerasClassifier(build_fn=build_model, verbose=1)
search = RandomizedSearchCV(model, hyperparameter, cv=3 )


ealystopping = EarlyStopping(monitor='val_loss',patience=30, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=30,
                            factor=0.5, verbose=1)

search.fit(x_train, y_train, validation_data=(x_val, y_val), 
        callbacks=[ealystopping, reduce_lr])

# search.save('./model/modelLoad/modelFolder/Dense_model1_11025sr.h5')

import pickle


# print("loss",loss)
bE = search.best_estimator_
loss, acc =bE.model.evaluate((x_test, y_test))
print('loss : ' ,loss)
print("acc: ",acc)

print('최적의 파라미터 : ', search.best_params_)
# pickle.dump(search.best_estimator_ ,open( './model/modelLoad/modelFolder/Dense_model1_wappingkeras_11025sr.pickle', 'wb'))
# pickle(search, open('./model/modelLoad/modelFolder/Dense_model1_wappingkeras_11025sr.dat', 'wb'))
bE.model.save('./test.h5')
# search.save('./test.h5')


# print(bE.model.evaluate((x_test, y_test))
'''
'''