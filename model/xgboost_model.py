from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
import pickle
import datetime
from sklearn import preprocessing

x = np.load('./npy/all_scale_x_final.npy')
y = np.load('./npy/all_scale_y_final.npy')


print(x[0])
print(y[0])
# from sklearn.utils import shuffle
y=preprocessing.LabelEncoder().fit_transform(y)
print(x[0])
print(len(np.unique(y)))
x_train, x_test ,y_train , y_test= train_test_split(x, y, train_size = 0.6)
x_val, x_test ,y_val , y_test= train_test_split(x_test, y_test, train_size = 0.5)



model = XGBClassifier(n_jobs=-1,
                predictor='gpu_predictor',
                tree_method='gpu_hist'
                 )

start_time = datetime.datetime.now()

model.fit(x_train, y_train, 
    early_stopping_rounds=40,
            eval_set=[(x_val, y_val)],
            verbose=True
            )
end_time = datetime.datetime.now()

model.score(x_test, y_test)
print(model.score(x_test, y_test))




print("총 데이터의 개수", x.shape[0])
print("score", model.score(x_test, y_test))
# print("내가 만든 wav는 48번인데 predict는???? ", y_predict)
print("fit 소요 시간",end_time - start_time)

pickle.dump(model, open('./model/modelLoad/modelFolder/xgboost_11025sr.dat', 'wb'))


'''
총 데이터의 개수 6084
score 0.8940016433853739
fit 소요 시간 0:00:10.090271
'''
