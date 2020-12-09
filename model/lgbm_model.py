from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import datetime
import pickle

x = np.load('./npy/all_scale_x.npy')
y = np.load('./npy/all_scale_y.npy')
x_predict = np.load('./npy/mag_tmp.npy')
print(x.shape)

x_predict = x_predict.reshape(1, x_predict.shape[0])

x_train, x_test ,y_train , y_test= train_test_split(x, y, train_size = 0.6)
x_val, x_test ,y_val , y_test= train_test_split(x_test, y_test, train_size = 0.5)

print(y)
model = LGBMClassifier(n_jobs=-1,
                     tree_method='gpu_hist',
                     predictor = 'gpu_predictor'
                     )
start_time = datetime.datetime.now()


model.fit(x_train, y_train, early_stopping_rounds=40,
            eval_set=(x_val, y_val),
            # verbose=True
            )

end_time = datetime.datetime.now()

model.score(x_test, y_test)


y_predict = model.predict(x_predict)

print(model.score(x_test, y_test))
print(y_predict)
print(end_time - start_time)

pickle.dump(model, open('./model/modelLoad/modelFolder/lgbm.dat', 'wb'))