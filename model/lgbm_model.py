from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import datetime
import pickle
from sklearn.model_selection import RandomizedSearchCV
x = np.load('./npy/all_scale_x_final.npy')
y = np.load('./npy/all_scale_y_final.npy')

print(x.shape)
parameters = [
    {
              'n_jobs' : [-1]
              } 

]

#train, test, val 을 구분하기 위한 train_test_split
x_train, x_test ,y_train , y_test= train_test_split(x, y, train_size = 0.6)
x_val, x_test ,y_val , y_test= train_test_split(x_test, y_test, train_size = 0.5)

model = LGBMClassifier()

search = RandomizedSearchCV(model, parameters, cv=3 )

search.fit(x_train, y_train, 
            early_stopping_rounds=40,
            eval_set=[(x_val, y_val)]
            )
from sklearn import metrics
print("총 데이터의 개수 : ", x.shape[0])
print("xtest : ", x_test.shape)
print("xtest : ", x_train.shape)
metrics.log_loss
y_pred = search.predict(x_test[4].reshape(1, 37))
print(y_pred)
y_pred = search.best_estimator_.predict(x_test)
print(y_pred)
print("score : ", search.best_estimator_.score(x_test, y_test))
print("score : ", search.best_estimator_.score(x_test, y_test))
print('best_param : ', search.best_params_)
# print("fit 소요 시간",end_time - start_time)

#lgbm 모델 저장.
pickle.dump(search.best_estimator_, open('./model/modelLoad/modelFolder/lgbm_11025sr_lgbml.dat', 'wb'))
'''
총 데이터의 개수 6084
score 0.8866064092029581
내가 만든 wav는 48번인데 predict는????  [48]
fit 소요 시간 0:00:02.726685
'''