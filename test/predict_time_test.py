from tensorflow.keras.models import load_model
import pickle
import numpy as np
import time

dnn_model = load_model('./model/modelLoad/modelFolder/Dense_model1_11025sr.h5')
lgbm_model = pickle.load(open('./model/modelLoad/modelFolder/lgbm_11025sr.dat', 'rb'))
xgb_model = pickle.load(open('./model/modelLoad/modelFolder/xgboost_11025sr.dat', 'rb'))

x_predict = np.load('./npy/mag_tmp.npy')

x_predict = x_predict.reshape(1,x_predict.shape[0])

dnn = []
xgb = []
lgbm = []

for i in range(100):
    start1 = time.time()
    dnn_model.predict(x_predict)
    end_time1 = time.time() -start1

    start2 = time.time()
    xgb_model.predict(x_predict)
    end_time2 = time.time() -start2

    start3 = time.time()
    lgbm_model.predict(x_predict)
    end_time3 = time.time() -start3

    dnn.append(end_time1)
    xgb.append(end_time2)
    lgbm.append(end_time3)


print('dnn 걸린 시간 100회평균', np.average(dnn))
print('xgb 걸린 시간 100회평균', np.average(xgb))
print('lgbm 걸린 시간 100회평균', np.average(lgbm))

'''
DNN fit 소요 시간:  0:01:44.247005
acc 0.8468112945556641

XGB fit 소요 시간 0:00:10.090271
score 0.8940016433853739

lgbm fit 소요 시간 0:00:02.726685
score 0.8866064092029581

dnn 걸린 시간 100회평균 0.11190287828445435
xgb 걸린 시간 100회평균 0.005963783264160156
lgbm 걸린 시간 100회평균 0.0010573649406433105
'''