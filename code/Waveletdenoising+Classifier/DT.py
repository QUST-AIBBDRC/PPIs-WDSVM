# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
#from time import time
import time
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import scale,StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import utils.tools as utils
start = time.time()
def lassodimension(data,label,alpha=np.array([0.005])):#alpha代表想要传递的alpha的一组值,用在循环中,来找出一个尽可能好的alpha的值
    lassocv=LassoCV(cv=5, alphas=alpha).fit(data, label)
#    lasso = Lasso(alpha=lassocv.alpha_, random_state=0)#生成Lasso对象
    x_lasso = lassocv.fit(data,label)#代入alpha进行降维
    mask = x_lasso.coef_ != 0 #mask是一个numpy数组,数组中的元素都是bool值,并且数组的维度和data的维度是相同的
    new_data = data[:,mask]  #将data中相应维度中与mask中为True对应的元素挑选出来
    return new_data,mask #返回降维之后的数组,并将使用lasso训练数据之后得到的最大值一起返回
data_train = sio.loadmat('Y.mat')
data=data_train.get('Y')
row=data.shape[0]
column=data.shape[1]
shu=data[:,np.array(range(1,column))]
shu=scale(shu)

a=int(row/2)
label1=np.ones((a,1))
label2=np.zeros((a,1))
label=np.append(label1,label2)

index = [i for i in range(row)]
np.random.shuffle(index)#shuffle the index
index=np.array(index)
X=shu[index,:]
y=label[index]
sepscores = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5
cv_clf = DecisionTreeClassifier(criterion="entropy")
skf= StratifiedKFold(n_splits=5)
for train, test in skf.split(X,y): 
    y_train=utils.to_categorical(y[train])
    hist=cv_clf.fit(X[train], y[train])
    #utils.plothistory(hist)
    #prediction probability
    y_score=cv_clf.predict_proba(X[test])
    yscore=np.vstack((yscore,y_score))
    y_test=utils.to_categorical(y[test]) 
    ytest=np.vstack((ytest,y_test))
    fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
    roc_auc = auc(fpr, tpr)
    y_class= utils.categorical_probas_to_classes(y_score)
    y_test_tmp=y[test]
    acc, precision,npv, sensitivity, specificity, mcc,f1 = utils.calculate_performace(len(y_class), y_class, y_test_tmp)
    sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
    print('NB:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))

result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)
result=sepscores
row=yscore.shape[0]
#column=data.shape[1]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('Y_X_sum_DT.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('Y_Y_sum_DT.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='NB ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.00])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('Y_DT_result.csv')
interval = (time.time() - start)
print("Time used:",interval)







