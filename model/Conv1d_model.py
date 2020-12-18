import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


x = np.load('./npy/all_scale_x_final.npy')
y = np.load('./npy/all_scale_y_final.npy')
y= y-48
y = to_categorical(y)
x = x.reshape(x.shape[0], x.shape[1], 1)


x_train, x_test ,y_train , y_test= train_test_split(x, y, train_size = 0.6)
x_val, x_test ,y_val , y_test= train_test_split(x_test, y_test, train_size = 0.5)

# print(x_test.shape)
# print(x_train.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, BatchNormalization
from tensorflow.keras.layers import Flatten, MaxPooling1D, Activation
from tensorflow.keras.callbacks import EarlyStopping

def crate_model(node1 = 128, node2 = 256, node3 = 256, node4 = 128,
             activation = 'relu', pool_size = 2,
             optimizer = Adam , padding='same', learning_rate = 0.001
                ):

    model = Sequential()
    model.add(Conv1D(node1, 3, padding=padding, input_shape=(x_train.shape[1],1)))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    
    model.add(Conv1D(node2, 3, padding=padding))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    
    model.add(Conv1D(node3, 3, padding=padding))
    model.add(Activation(activation))
    model.add(BatchNormalization())
   
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Activation(activation))
    model.add(Flatten())
    
    model.add(Dense(node4 ))
    model.add(Activation(activation))
    model.add(BatchNormalization())
    model.add(Dense(37, activation='softmax')) 
    model.compile(loss='categorical_crossentropy', 
                            metrics=['acc'], 
                            optimizer=optimizer(learning_rate = learning_rate))
    return model 
ealystopping = EarlyStopping(monitor='val_loss',
                            patience=10,
                            mode='auto')

def create_param():
    
    node1 = [32,64,128,512]
    node2 = [32,64,128,512]
    node3 = [32,64,128,512]
    node4 = [32,64,128,512]
    activation = ['relu', 'elu', 'selu']
    pool_size = [2,3,4]
    optimizer = [Adam, RMSprop, Adagrad]
    lr = [0.001, 0.003, 0.005, 0.007,0.0008, 0.009]
    
    return {
        'node1' : node1, 'node2' : node2, 'node3' : node3, 'node4' : node4,
        'activation': activation, 'pool_size' : pool_size, 'optimizer' : optimizer,
        'learning_rate' : lr
    }
hyperparam = create_param()

model = KerasClassifier(build_fn=crate_model, verbose=1)
search = RandomizedSearchCV(model, hyperparam, cv=3 )

search.fit(x_train, y_train, 
                    # epochs=1, 
                    epochs=100, 
                    batch_size=512, 
                    validation_data=(x_val, y_val), 
                    callbacks=[ealystopping])


# loss, acc =be.evaluate(x_test, y_test, batch_size=512)

# predic = search.predict(x_test) #됨
# predic = search.best_estimator_.predict(x_test) #됨
# predic = search.best_estimator_.model.predict(x_test) #됨
# predic = search.best_estimator_.model.evaluate(x_test, y_test, batc_size= 512) #됨

# print(predic.shape)
bE = search.best_estimator_

# loss, acc =bE.model.evaluate((x_test, y_test))
# print('loss : ' ,loss)
# print("acc: ",acc)
# print('acc : ',acc)
print('param : ', search.best_params_)
# search.best_estimator_.model.save('./model/modelLoad/modelFolder/covn1D_parma_tmp.h5')

# 5. 모델 학습 과정 표시하기

