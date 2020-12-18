import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from lightgbm import LGBMClassifier
import time
import pickle

RANDOM_STATE = 44

x = np.load('./npy/all_scale_x_final.npy')
y = np.load('./npy/all_scale_y_final.npy')


#train, test, val 을 구분하기 위한 train_test_split
x_train, x_test ,y_train , y_test= train_test_split(
    x, y, train_size = 0.6, random_state=RANDOM_STATE)
x_test, x_val ,y_test , y_val= train_test_split(
    x_test, y_test, train_size = 0.5, random_state=RANDOM_STATE)

print('x_train.shape:',x_train.shape)
print('x_val.shape:',x_val.shape)
print('x_test.shape:',x_test.shape)

start_time = time.time()

def model(
    num_leaves,
    max_depth,
    learning_rate,
    n_estimators):
    model = LGBMClassifier(n_jobs=6,
            num_leaves = int(num_leaves),
            max_depth = int(max_depth),
            learning_rate = learning_rate,
            n_estimators = int(n_estimators)
            )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("score", acc)
    return acc


from sklearn.model_selection import RandomizedSearchCV
parameters = {
    'num_leaves':np.array(range(16,1024,1)), # (16, 1024), 트리 리프의 수
    'max_depth':np.array(range(7,20,1)), # (7, 20), 트리의 최대 깊이
    'learning_rate':np.arange(0.001,0.1,0.01), # (0.001, 0.1), 학습률
    'n_estimators':np.array(range(50,200,1)), # (50, 200), 생성할 트리의개수
}
start_time = time.time()
RSCV = RandomizedSearchCV(LGBMClassifier(metrics=['multi_logloss']), 
                        parameters, cv=5,
                        random_state=RANDOM_STATE,
                        verbose=2)
RSCV.fit(x_train,y_train, eval_set=[(x_val, y_val)])

score = RSCV.score(x_test, y_test)
print("score:", score)
best_params = RSCV.best_params_
print("최적의 파라미터:", best_params)
print("RSCV fit 소요 시간: %.3fs" %((time.time() - start_time)) )

pickle.dump(best_params, open('./model/modelLoad/modelFolder/lgbm_11025sr_lgbml.dat', 'wb'))

best_model = RSCV.best_estimator_
score = best_model.score(x_test, y_test)
print("best_model score:", score)

'''
'''

