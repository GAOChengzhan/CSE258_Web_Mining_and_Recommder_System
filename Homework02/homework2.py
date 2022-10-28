import pandas as pd
import numpy as np
import random
from collections import defaultdict
import warnings
import statistics
warnings.filterwarnings('ignore')
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N
#store the answer
answers = {}
#read the data from file'5year.arff'
f = open("data/5year.arff", 'r')
# Read and parse the data
while not '@data' in f.readline():
    pass

dataset = []
for l in f:
    if '?' in l: # Missing entry
        continue
    l = l.split(',')
    values = [1] + [float(x) for x in l]
    values[-1] = values[-1] > 0 # Convert to bool
    dataset.append(values)
#Task1----------------------------------------------------------------------------
#build logistic model
datasetNP=np.array(dataset)
dataX=datasetNP[:,:-1]
dataY=datasetNP[:,-1]
from sklearn.linear_model import LogisticRegression
model_Task1=LogisticRegression(C=1.0)
model_Task1.fit(dataX,dataY)
#calculate the accuracy and Balanced Error Rate (BER)
def accuracy_BER_calculation(y,y_predict):
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
    accuracy=(TrueNg+TruePt)/len(y)
    return [accuracy,BER]
dataY_pre1=model_Task1.predict(dataX)
answers['Q1']= accuracy_BER_calculation(dataY,dataY_pre1)
assertFloatList(answers['Q1'], 2)
#Task2----------------------------------------------------------------------------
model_Task2=LogisticRegression(C=1.0,class_weight='balanced')
model_Task2.fit(dataX,dataY)
dataY_pre2=model_Task2.predict(dataX)
answers['Q2']= accuracy_BER_calculation(dataY,dataY_pre2)
assertFloatList(answers['Q2'], 2)
#Task3----------------------------------------------------------------------------
random.seed(3)
random.shuffle(dataset)
datasetNP3=np.array(dataset)
dataX3=datasetNP3[:,:-1]
dataY3=datasetNP3[:,-1]
length=len(dataX3)
Xtrain, Xvalid, Xtest = dataX3[:length//2], dataX3[length//2:(3*length)//4], dataX3[(3*length)//4:]
ytrain, yvalid, ytest = dataY3[:length//2], dataY3[length//2:(3*length)//4], dataY3[(3*length)//4:]
model_Task3=LogisticRegression(C=1.0,class_weight='balanced')
model_Task3.fit(Xtrain,ytrain)
ytrain_pre=model_Task3.predict(Xtrain)
yvalid_pre=model_Task3.predict(Xvalid)
ytest_pre=model_Task3.predict(Xtest)
[berTrain, berValid, berTest]=[accuracy_BER_calculation(ytrain,ytrain_pre)[1],\
                               accuracy_BER_calculation(yvalid,yvalid_pre)[1],\
                               accuracy_BER_calculation(ytest,ytest_pre)[1]]
answers['Q3'] = [berTrain, berValid, berTest]
assertFloatList(answers['Q3'], 3)
#Task4----------------------------------------------------------------------------
C_val=[10**(-4),10**(-3),10**(-2),10**(-1),1,10,10**2,10**3,10**4]
#Report the validation BER
berList=[]
for c in C_val:
    model_Task4=LogisticRegression(C=c,class_weight='balanced')
    model_Task4.fit(Xtrain,ytrain)
    yvalid_pre=model_Task4.predict(Xvalid)
    berList.append(accuracy_BER_calculation(yvalid,yvalid_pre)[1])
answers['Q4'] = berList
assertFloatList(answers['Q4'], 9)
#Task5----------------------------------------------------------------------------
berList5=[]
for c in C_val:
    model_Task5=LogisticRegression(C=c,class_weight='balanced')
    model_Task5.fit(Xtrain,ytrain)
    ytest_pre=model_Task5.predict(Xtest)
    berList5.append(accuracy_BER_calculation(ytest,ytest_pre)[1])
bestC=10
ber5=berList5[5]
answers['Q5'] = [bestC, ber5]
assertFloatList(answers['Q5'], 2)

#Task6----------------------------------------------------------------------------
data_taskRecommend=pd.read_json('data/young_adult_10000.json', lines=True)
dataTrain = data_taskRecommend[:9000]
dataTest = data_taskRecommend[9000:]
usersPerBook = defaultdict(set)
booksPerUser = defaultdict(set)
reviewsPerUser = defaultdict(list)
reviewsPerBook = defaultdict(list)
globalAvg=0
trainRatingDict = {} # To retrieve a rating for a specific user/item pair
for i in range(len(dataTrain)):
    d=dataTrain.iloc[i,:]
    user,book = d['user_id'], d['book_id']
    rating=d['rating']
    globalAvg+=rating
    usersPerBook[book].add(user)
    booksPerUser[user].add(book)
    reviewsPerUser[user].append(rating)
    reviewsPerBook[book].append(rating)
    trainRatingDict[(user,book)]=rating
globalAvg=globalAvg/len(dataTrain)
def JaccardSimilarity(set1,set2):
    inter = len(set1.intersection(set2))
    uni = len(set1.union(set2))
    return inter/uni
def mostKSimilar (targetedBook, K): 
    JSimList = []
    users = usersPerBook[targetedBook] 
    for comparedBook in usersPerBook : 
        if targetedBook == comparedBook: 
            continue
        similarity = JaccardSimilarity(users,usersPerBook[comparedBook])
        JSimList.append((similarity ,comparedBook))
    JSimList.sort(key=lambda x:x[0],reverse=True) # Sort to find the most similar
    return JSimList[:K]
answers['Q6'] = mostKSimilar(2767052, 10)
assert len(answers['Q6']) == 10
assertFloatList([x[0] for x in answers['Q6']], 10)
#Task7----------------------------------------------------------------------------
avgRatingPerBook=defaultdict(float)
for book in reviewsPerBook:
    ReviewList=reviewsPerBook[book]
    avgRatingPerBook[book]=statistics.mean(ReviewList)
#testset
testRatingDict = {} 
for i in range(len(dataTest)):
    d=dataTest.iloc[i,:]
    user,book = d['user_id'], d['book_id']
    rating=d['rating']
    testRatingDict[(user,book)]=rating

#calculate global average ratings of all books
predictUserBook=[]
sumSim=[]
for (user,book) in testRatingDict:
    numerator=0
    denominator=0
    users= usersPerBook[book] 
    for comparedBook in booksPerUser[user]:
        sim_ij=0
        if book == comparedBook: 
            continue
        sim_ij = JaccardSimilarity(users,usersPerBook[comparedBook])
        denominator+=(sim_ij)
        numerator+=((trainRatingDict[(user,comparedBook)]-avgRatingPerBook[comparedBook])*sim_ij)
    sumSim.append(denominator)
    if  denominator!=0:
        ans=(avgRatingPerBook[book]+(numerator/denominator))  
    else:
        ans= globalAvg
    predictUserBook.append(ans)
#calculate MSE
SquaredError=0
for i,(user,book) in enumerate(testRatingDict):
    Error=testRatingDict[(user,book)]-predictUserBook[i]
    SquaredError+=(Error**2)
mse7=SquaredError/(len(testRatingDict))
answers['Q7'] = mse7
assertFloat(answers['Q7'])

#Task8----------------------------------------------------------------------------
avgRatingPerUser=defaultdict(float)
for user in reviewsPerUser:
    ReviewList=reviewsPerUser[user]
    avgRatingPerUser[user]=statistics.mean(ReviewList)
predictUserBook=[]
for (user,book) in testRatingDict:
    numerator=0
    denominator=0
    books = booksPerUser[user] 
    for comparedUser in usersPerBook[book]:
        if user == comparedUser: 
            continue
        sim_uv = JaccardSimilarity(books,booksPerUser[comparedUser])
        denominator+=(sim_uv)
        numerator+=((trainRatingDict[(comparedUser,book)]-avgRatingPerUser[comparedUser])*sim_uv)
    if  denominator!=0:
        ans=(avgRatingPerUser[user]+(numerator/denominator))  
    else:
        ans= globalAvg
    predictUserBook.append(ans)
#calculate MSE
SquaredError=0
for i,(user,book) in enumerate(testRatingDict):
    Error=testRatingDict[(user,book)]-predictUserBook[i]
    SquaredError+=Error**2
mse8=SquaredError/(len(testRatingDict))
answers['Q8'] = mse8
assertFloat(answers['Q8'])
f = open("answers_hw2.txt", 'w')
f.write(str(answers) + '\n')