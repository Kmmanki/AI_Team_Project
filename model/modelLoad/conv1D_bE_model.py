from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


x = np.load('./npy/all_scale_x_final.npy')
y = np.load('./npy/all_scale_y_final.npy')
y= y-48
y = to_categorical(y)
x = x.reshape(x.shape[0], x.shape[1], 1)


x_train, x_test ,y_train , y_test= train_test_split(x, y, train_size = 0.6)
x_val, x_test ,y_val , y_test= train_test_split(x_test, y_test, train_size = 0.5)

model = load_model('./model/modelLoad/modelFolder/covn1D_parma_tmp.h5')
loss, acc = model.evaluate(x_test, y_test)
print('acc : ',acc)

print('loss : ', loss)