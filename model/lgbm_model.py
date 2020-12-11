from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import datetime
import pickle

x = np.load('./npy/all_scale_x_11025sr.npy')
y = np.load('./npy/all_scale_y_11025sr.npy')
x_predict = np.load('./npy/mag_tmp.npy')

print(x.shape)

x_predict = x_predict.reshape(1, x_predict.shape[0])

#train, test, val 을 구분하기 위한 train_test_split
x_train, x_test ,y_train , y_test= train_test_split(x, y, train_size = 0.6)
x_val, x_test ,y_val , y_test= train_test_split(x_test, y_test, train_size = 0.5)

model = LGBMClassifier(n_jobs=-1,
                     tree_method='gpu_hist',
                     predictor = 'gpu_predictor'
                     )

start_time = datetime.datetime.now()
model.fit(x_train, y_train, early_stopping_rounds=40,
            eval_set=[(x_val, y_val)],
            # verbose=True
            )
end_time = datetime.datetime.now()

y_predict = model.predict(x_predict)

print("총 데이터의 개수", x.shape[0])
print("score", model.score(x_test, y_test))
print("내가 만든 wav는 48번인데 predict는???? ", y_predict)
print("fit 소요 시간",end_time - start_time)

#lgbm 모델 저장.
pickle.dump(model, open('./model/modelLoad/modelFolder/lgbm_11025sr.dat', 'wb'))
'''
총 데이터의 개수 6084
score 0.8866064092029581
내가 만든 wav는 48번인데 predict는????  [48]
fit 소요 시간 0:00:02.726685
'''