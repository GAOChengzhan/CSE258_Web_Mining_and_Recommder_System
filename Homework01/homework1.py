import sklearn as sk
import pandas as pd
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
data=pd.read_json('young_adult_10000.json', lines=True)
data_task1=data[['rating','review_text']]
#Task1-------------------------------------------------------------------------
data_task1['Count_of_Exclamation']=data_task1['review_text'].str.count('!')
data_task1_new=data_task1[['rating','Count_of_Exclamation']]
from sklearn import linear_model
def trainModel(model,X_train,y_train):
    model.fit(X_train,y_train)
    W=model.coef_
    b=model.intercept_
    return W,b
LR_model1=linear_model.LinearRegression(fit_intercept=True)
X1=np.array(data_task1_new['Count_of_Exclamation']).reshape(-1,1)
y1=np.array(data_task1_new['rating'])
W1,b1=trainModel(LR_model1,X1,y1)
print(W1,b1)
#calculate MSE
def MSE_calculation(LR_model,X,y,num):
    y_pred=LR_model.predict(X)
    SE=[i**2 for i in (y-y_pred)]
    SSE=sum(SE)
    MSE=SSE/len(y)
    print("The Task{}'s MSE is {:.5f}".format(num,MSE))
MSE_calculation(LR_model1,X1,y1,1)
#Task2-------------------------------------------------------------------------
data_task2=data[['rating','review_text']]
data_task2['Count_of_Exclamation']=data_task2['review_text'].str.count('!')
data_task2['Len_of_Review']=data_task2['review_text'].str.len()
data_task2_new=data_task2[['rating','Len_of_Review','Count_of_Exclamation']]
X2=np.array(data_task2_new[['Len_of_Review','Count_of_Exclamation']])
y2=np.array(data_task2_new['rating'])
LR_model2=linear_model.LinearRegression(fit_intercept=True)
W2,b2=trainModel(LR_model2,X2,y2)
print(W2,b2)
MSE_calculation(LR_model2,X2,y2,2)
#Task3-------------------------------------------------------------------------
data_task3=data_task1_new[:]
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
def train_PolyModel(degree,X_train,y_train):
    PolyRegression=make_pipeline(PolynomialFeatures(degree=degree,include_bias=True),\
                                 linear_model.LinearRegression(fit_intercept=True))
    PolyRegression.fit(X_train,y_train)
    return PolyRegression
def poly_plot(polyModel,X,y,X_predict):
    plt.figure()
    plt.scatter(X,y)
    y_predict=polyModel.predict(X_predict)
    plt.plot(X_predict,y_predict,c="g")
    plt.title("Task3")
    plt.show()
def MSE_calculation(LR_model,X,y):
    y_pred=LR_model.predict(X)
    SE=[i**2 for i in (y-y_pred)]
    SSE=sum(SE)
    MSE=SSE/len(y)
    return MSE
X3=np.array(data_task3['Count_of_Exclamation']).reshape(-1,1)
y3=np.array(data_task3['rating'])
x_predict3=np.arange(0,60,0.1)
MSE_lst=[]
for i in range(1,6):
    PolyRegression=train_PolyModel(i,X3,y3)
    poly_plot(PolyRegression,X3,y3,x_predict3.reshape(-1,1))
    MSE_lst.append(MSE_calculation(PolyRegression,X3,y3))
print(MSE_lst)
#Task4-------------------------------------------------------------------------
half=len(data_task3)//2
X4_train=np.array(data_task3[:half]['Count_of_Exclamation']).reshape(-1,1)
y4_train=np.array(data_task3[:half]['rating'])
X4_test=np.array(data_task3[half:]['Count_of_Exclamation']).reshape(-1,1)
y4_test=np.array(data_task3[half:]['rating'])
MSE_lst_task4=[]
for i in range(1,6):
    PolyRegression=train_PolyModel(i,X4_train,y4_train)
    MSE_lst_task4.append(MSE_calculation(PolyRegression,X4_test,y4_test))
print("MSE with different on test set:")
print(MSE_lst_task4)
#Task5-------------------------------------------------------------------------
# using median as predictor is one best way to predict the value with MAE loss
from statistics import median
y5_train_median=median(y4_train)
print("MAE on the test set of Task4:")
MAE=sum([abs(i-y5_train_median) for i in y4_test])/len(y4_test)
print(MAE)

#Task6-------------------------------------------------------------------------
def parseData(fname):
    for l in open(fname):
        yield eval(l)
data2 = list(parseData("beer_50000.json"))
print(data2[0])
#discarding entries that don’t include a specified gender
data2_gender=[d for d in data2 if 'user/gender' in d]
data_task6=np.array([[ 0 if d['user/gender'] =='Male' else 1, d['review/text'].count('!')] for d in data2_gender])
X_task6=data_task6[:,1].reshape(-1,1)
y_task6=data_task6[:,0]
from sklearn.linear_model import LogisticRegression
LgR_model_task6=LogisticRegression()
LgR_model_task6.fit(X_task6,y_task6)
y_predict_task6=LgR_model_task6.predict(X_task6)
def accuracy_calculation(y,y_predict):
    TruePt,TrueNg,FalsePt,FalseNg=0,0,0,0
    for i in range(len(y_predict)):
        if y_predict[i]==0:
            if y[i]==0:
                TrueNg+=1
            else:
                FalseNg+=1
        else:
            if y[i]==1:
                TruePt+=1
            else:
                FalsePt+=1
    BER=0.5*((FalsePt/(FalsePt+TrueNg))+(FalseNg/(FalseNg+TruePt)))
    return [TruePt,TrueNg,FalsePt,FalseNg,BER]
res_task6 = accuracy_calculation(y_task6,y_predict_task6)
print(res_task6)
#Task7-------------------------------------------------------------------------
LgR_model_task7=LogisticRegression(class_weight='balanced')
LgR_model_task7.fit(X_task6.reshape(-1,1),y_task6)
y_predict_task7=LgR_model_task7.predict(X_task6)
res_task7 = accuracy_calculation(y_task6,y_predict_task7)
print(res_task7)
#Task8-------------------------------------------------------------------------
y_predict_task8 = LgR_model_task7.predict_proba(X_task6)
y_predict_task8new=[]
for i,j in enumerate(y_predict_task8):
    y_predict_task8new.append([j[1],i])
y_predict_task8new.sort(reverse=True)
def precisionAtK(y,y_predict,k):
    TruePt,TrueNg,FalsePt,FalseNg=0,0,0,0
    for i in range(k):
        if y_predict[i][0]>0.50:
            if y[y_predict[i][1]]==1:
                TruePt+=1
            else:
                FalsePt+=1
        else:
            if y[y_predict[i][1]]==0:
                TrueNg+=1
            else:
                FalseNg+=1
            
    return (TruePt/(TruePt+FalsePt))
res=[]
for k in [1, 10, 100, 1000, 10000]:
    res.append(precisionAtK(y_task6,y_predict_task8new,k))
print(res)