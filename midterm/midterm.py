import json
import gzip
import math
from collections import defaultdict
import numpy
from sklearn import linear_model

# This will suppress any warnings, comment out if you'd like to preserve them
import warnings
warnings.filterwarnings("ignore")

# Check formatting of submissions
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N
    
# Read data
answers = {}
f = open("spoilers.json.gz", 'r')
dataset = []
for l in f:
    d = eval(l)
    dataset.append(d)
f.close()

# A few utility data structures
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)

for d in dataset:
    u,i = d['user_id'],d['book_id']
    reviewsPerUser[u].append(d)
    reviewsPerItem[i].append(d)

# Sort reviews per user by timestamp
for u in reviewsPerUser:
    reviewsPerUser[u].sort(key=lambda x: x['timestamp'])
    
# Same for reviews per item
for i in reviewsPerItem:
    reviewsPerItem[i].sort(key=lambda x: x['timestamp'])
### 1a ##############################################################################
# MSE function
def MSE(y, ypred):
    return sum([(a-b)**2 for (a,b) in zip(y,ypred)]) / len(y)
y=[]
ypred=[]
for u in reviewsPerUser:
    if len(reviewsPerUser[u])<=1:#skip those only have one review
        continue
    TobePred=reviewsPerUser[u][-1]
    PreviousPred=reviewsPerUser[u][:-1]
    y.append(TobePred['rating'])
    sumRating=0
    for pp in PreviousPred:
        sumRating+=pp['rating']
    avgRating=sumRating/len(PreviousPred)
    ypred.append(avgRating)
MSE1a=MSE(y,ypred)
answers['Q1a'] = MSE1a
assertFloat(answers['Q1a'])   
print("MSE1a:{}".format(MSE1a))

### 1b ##############################################################################
y=[]
ypred=[]
for i in reviewsPerItem:
    if len(reviewsPerItem[i])<=1:#skip those only have one review
        continue
    TobePred=reviewsPerItem[i][-1]
    PreviousPred=reviewsPerItem[i][:-1]
    y.append(TobePred['rating'])
    sumRating=0
    for pp in PreviousPred:
        sumRating+=pp['rating']
    avgRating=sumRating/len(PreviousPred)
    ypred.append(avgRating)

MSE1b=MSE(y,ypred)
answers['Q1b'] = MSE1b
assertFloat(answers['Q1b'])
print("MSE1b:{}".format(MSE1b))
### 2 ###############################################################################
answers['Q2'] = []
for N in [1,2,3]:
    #initialize
    y=[]
    ypred=[]
    for u in reviewsPerUser:
        if len(reviewsPerUser[u])<=1: #skip those who have less than N+1 reviews
            continue
        if len(reviewsPerUser[u])< N+1:
            TobePred=reviewsPerUser[u][-1]
            PreviousPred=reviewsPerUser[u][:-1]
            y.append(TobePred['rating'])
            sumRating=0
            for pp in PreviousPred:
                sumRating+=pp['rating']
            avgRating=sumRating/len(PreviousPred)
            ypred.append(avgRating)
            
        else:
            TobePred=reviewsPerUser[u][-1]
            PreviousPred=reviewsPerUser[u][-(N+1):-1]
            y.append(TobePred['rating'])

            sumRating=0
            for pp in PreviousPred:
                sumRating+=pp['rating']
            avgRating=sumRating/len(PreviousPred)
            ypred.append(avgRating)
    mse=MSE(y,ypred)
    answers['Q2'].append(mse)
assertFloatList(answers['Q2'], 3)
print("answers['Q2']:{}".format(answers['Q2']))
### 3 ###############################################################################
### 3a ##############################################################################

def feature3(N, u): # For a user u and a window size of N
    ratings=reviewsPerUser[u]
    if len(ratings)<=N+1:
        return None
    temp=[]
    for r in ratings[-N-1:-1]:
        temp.append(r['rating'])
    ans=[1]
    ans.extend(temp[::-1])
    return ans
answers['Q3a'] = [feature3(2,dataset[0]['user_id']), feature3(3,dataset[0]['user_id'])]
assert len(answers['Q3a']) == 2
assert len(answers['Q3a'][0]) == 3
assert len(answers['Q3a'][1]) == 4
print("answers['Q3a']:{}".format(answers['Q3a']))

### 3b ##############################################################################
answers['Q3b'] = []
for N in [1,2,3]:
    Xtrain = []
    ytrain = []
    for u in reviewsPerUser:
        if len(reviewsPerUser[u])<=N+1:
            continue
        feature=feature3(N, u)
        TobePred=reviewsPerUser[u][-1]
        ytrain.append(TobePred['rating'])
        Xtrain.append(feature)
    mod3 = linear_model.ARDRegression(fit_intercept=False)
    mod3.fit(Xtrain,ytrain)
    ypred = mod3.predict(Xtrain)
    mse=MSE(ytrain,ypred)      
    answers['Q3b'].append(mse)
assertFloatList(answers['Q3b'], 3)
print("answers['Q3b']:{}".format(answers['Q3b']))
### 4 ###############################################################################
globalAverage = [d['rating'] for d in dataset]
globalAverage = sum(globalAverage) / len(globalAverage)
### 4a ##############################################################################
def featureMeanValue(N, u): # For a user u and a window size of N
    ratings=reviewsPerUser[u]
    missingNum=(N+1)-len(ratings)
    ans=[1]
    if missingNum==N:
        ans.extend([globalAverage]*missingNum)
        return ans
    elif missingNum<=0:
        temp=[]
        for r in ratings[-(N+1):-1]:
            temp.append(r['rating'])
        ans.extend(temp[::-1])
        return ans
    else:
        temp=[]
        for r in ratings[:-1]:
            temp.append(r['rating'])
        localAvg=sum(temp)/len(temp)
        ans.extend(temp[::-1])
        ans.extend([localAvg]*missingNum)
        return ans    
    
def featureMissingValue(N, u):
    ratings=reviewsPerUser[u]
    missingNum=(N+1)-len(ratings)
    ans=[1]
    if missingNum==N:
        for i in range(missingNum):
            ans.extend([1,0])
        return ans
    elif missingNum<=0:
        temp=ratings[-(N+1):-1]
        temp=temp[::-1]
        for t in temp:
            ans.extend([0,t['rating']])
        return ans
    else:
        temp=ratings[:-1]
        temp=temp[::-1]
        for t in temp:
            ans.extend([0,t['rating']])
        for i in range(missingNum):
            ans.extend([1,0])
        return ans
    
answers['Q4a'] = [featureMeanValue(10, dataset[0]['user_id']), featureMissingValue(10, dataset[0]['user_id'])]
assert len(answers['Q4a']) == 2
assert len(answers['Q4a'][0]) == 11
assert len(answers['Q4a'][1]) == 21
print("answers['Q4a']:{}".format(answers['Q4a'] ))
### 4b ##############################################################################
answers['Q4b'] = []
N=10
for featFunc in [featureMeanValue, featureMissingValue]:
    Xtrain = []
    ytrain = []
    for u in reviewsPerUser:
        if len(reviewsPerUser[u])<=0:
            continue
        feature=featFunc(N, u)
        TobePred=reviewsPerUser[u][-1]
        ytrain.append(TobePred['rating'])
        Xtrain.append(feature)
    mod4 = linear_model.ARDRegression(fit_intercept=False)
    mod4.fit(Xtrain,ytrain)
    ypred = mod4.predict(Xtrain)
    mse=MSE(ytrain,ypred)      
    answers['Q4b'].append(mse)
assertFloatList(answers["Q4b"], 2)
print("answers['Q4b']:{}".format(answers['Q4b'] ))
### 5 ###############################################################################
def feature5(sentence):
    ans=[1]
    lenChar=len(sentence)
    countExclamation=0
    numCapital=0
    for c in sentence:
        if c=='!':
            countExclamation+=1
        if c.isupper():
            numCapital+=1
    ans.extend([lenChar,countExclamation,numCapital])
    return ans
y = []
X = []
for d in dataset:
    for spoiler,sentence in d['review_sentences']:
        X.append(feature5(sentence))
        y.append(spoiler)
mod5 = linear_model.LogisticRegression(class_weight='balanced',C=1)
mod5.fit(X,y)
y_predict=mod5.predict(X)
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
answers['Q5a'] = X[0]
answers['Q5b'] = accuracy_calculation(y,y_predict)
assert len(answers['Q5a']) == 4
assertFloatList(answers['Q5b'], 5)
print("answers['Q5a']:{}".format(answers['Q5a'] ))
print("answers['Q5b']:{}".format(answers['Q5b'] ))
### 6 ###############################################################################
def feature6(review):
    sentences = review['review_sentences']
    sentence=sentences[5][1]
    ans=[1]
    lenChar=len(sentence)
    countExclamation=0
    numCapital=0
    for c in sentence:
        if c=='!':
            countExclamation+=1
        if c.isupper():
            numCapital+=1
    ans.extend([lenChar,countExclamation,numCapital])

    for i in range(5):
        ans.append(sentences[i][0])
    return ans
    
y = []
X = []

for d in dataset:
    sentences = d['review_sentences']
    if len(sentences) < 6: continue
    X.append(feature6(d))
    y.append(sentences[5][0])

mod6 = linear_model.LogisticRegression(class_weight='balanced',C=1)
mod6.fit(X,y)
y_predict=mod6.predict(X)
answers['Q6a'] = X[0]
answers['Q6b'] = accuracy_calculation(y,y_predict)[-1]
assert len(answers['Q6a']) == 9
assertFloat(answers['Q6b'])
print("answers['Q6a']:{}".format(answers['Q6a'] ))
print("answers['Q6b']:{}".format(answers['Q6b'] ))
### 7 ###############################################################################
y = []
X = []

for d in dataset:
    sentences = d['review_sentences']
    if len(sentences) < 6: continue
    X.append(feature6(d))
    y.append(sentences[5][0])
# 50/25/25% train/valid/test split
Xtrain, Xvalid, Xtest = X[:len(X)//2], X[len(X)//2:(3*len(X))//4], X[(3*len(X))//4:]
ytrain, yvalid, ytest = y[:len(X)//2], y[len(X)//2:(3*len(X))//4], y[(3*len(X))//4:]
bers=[]
for c in [0.01, 0.1, 1, 10, 100]: 
    mod7 = linear_model.LogisticRegression(class_weight='balanced',C=c)
    mod7.fit(Xtrain,ytrain)
    yvalid_predict=mod7.predict(Xvalid)
    bers.append(accuracy_calculation(yvalid,yvalid_predict)[-1])
print("bers:{}".format(bers))
bestC=0.1
mod7 = linear_model.LogisticRegression(class_weight='balanced',C=bestC)
mod7.fit(Xtrain,ytrain)
ytest_predict=mod7.predict(Xtest)
ber=accuracy_calculation(ytest,ytest_predict)[-1]
answers['Q7'] = bers + [bestC] + [ber]
assertFloatList(answers['Q7'], 7)
print("answers['Q7']:{}".format(answers['Q7']))
### 8 ###############################################################################
#Jaccard Similarity
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom

# 75/25% train/test split
dataTrain = dataset[:15000]
dataTest = dataset[15000:]

# A few utilities

itemAverages = defaultdict(list)
ratingMean = []

for d in dataTrain:
    itemAverages[d['book_id']].append(d['rating'])
    ratingMean.append(d['rating'])

for i in itemAverages:
    itemAverages[i] = sum(itemAverages[i]) / len(itemAverages[i])

ratingMean = sum(ratingMean) / len(ratingMean)

reviewsPerUser = defaultdict(list)
usersPerItem = defaultdict(set)

for d in dataTrain:
    u,i = d['user_id'], d['book_id']
    reviewsPerUser[u].append(d)
    usersPerItem[i].add(u)

# From my HW2 solution, welcome to reuse
def predictRating(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            return ratingMean
predictions=[]
labels=[]
for d in dataTest:
    user,item=d['user_id'],d['book_id']
    predict=predictRating(user,item)
    predictions.append(predict)
    labels.append(d['rating'])
answers["Q8"] = MSE(predictions, labels)
assertFloat(answers["Q8"])
print("answers['Q8']:{}".format(answers["Q8"]))
### 9 ###############################################################################
dataTest1=[]
dataTest2=[]
dataTest3=[]
for d in dataTest:
    user,item=d['user_id'],d['book_id']
    if item not in itemAverages:
        dataTest1.append(d)
    else:
        if len(usersPerItem[item])>5:
            dataTest3.append(d)
        else:
            dataTest2.append(d)
        
print("dataset1:{}".format(len(dataTest1)))
print("dataset2:{}".format(len(dataTest2)))
print("dataset3:{}".format(len(dataTest3)))
print("dataTest={}=dataset1+dataset2+dataset3={}".format(len(dataTest),\
                    sum([len(dataTest1),len(dataTest2),len(dataTest3)])))
mseLst=[]
for dataSet in [dataTest1,dataTest2,dataTest3]:
    predictions=[]
    labels=[]
    for d in dataSet:
        user,item=d['user_id'],d['book_id']
        predict=predictRating(user,item)
        predictions.append(predict)
        labels.append(d['rating'])
    mseLst.append(MSE(predictions, labels))
answers["Q9"] = mseLst
assertFloatList(answers["Q9"], 3)
print("answers['Q9']:{}".format(answers["Q9"]))
### 10 ##############################################################################
userAverages = defaultdict(list)
for d in dataTrain:
    userAverages[d['user_id']].append(d['rating'])
for i in userAverages:
    userAverages[i] = sum(userAverages[i]) / len(userAverages[i])

def predictRating10(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d['book_id']
        if i2 == item: continue
        ratings.append(d['rating'] - itemAverages[i2])
        similarities.append(Jaccard(usersPerItem[item],usersPerItem[i2]))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        if item in itemAverages:
            return itemAverages[item]
        else:
            if user in userAverages:
                return userAverages[user]
            return ratingMean
predictions=[]
labels=[]
for d in dataTest1:
    user,item=d['user_id'],d['book_id']
    predict=predictRating10(user,item)
    predictions.append(predict)
    labels.append(d['rating'])
itsMSE=MSE(predictions, labels)
description="Build another dictionary called 'userAverages' which is similar to\
            'itemAverages' and records the averages of ratings per user, \
            who have appeared in trainDataSet. When finding the items has not \
            appeared in trainData, we turn to 'userAverages'. If the user has \
            appeared in trainData, then we use the average of this user's ratings \
            to predict the rating. If neither user nor the item has appeared in \
            the trainData, we can only return the global average of ratings."
answers["Q10"] = (description, itsMSE)
assert type(answers["Q10"][0]) == str
assertFloat(answers["Q10"][1])
f = open("answers_midterm.txt", 'w')
f.write(str(answers) + '\n')
f.close()