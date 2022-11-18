import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import warnings
warnings.filterwarnings("ignore")
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)
def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r
answers = {}
### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before

allRatings = []
userRatings = defaultdict(list)

for user,book,r in readCSV("train_Interactions.csv.gz"):
    r = int(r)
    allRatings.append(r)
    userRatings[user].append(r)

globalAverage = sum(allRatings) / len(allRatings)
userAverage = {}
for u in userRatings:
    userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

predictions = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"):
    #header
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    if u in userAverage:
        predictions.write(u + ',' + b + ',' + str(userAverage[u]) + '\n')
    else:
        predictions.write(u + ',' + b + ',' + str(globalAverage) + '\n')

predictions.close()

### Would-read baseline: just rank which books are popular and which are not, and return '1' if a book is among the top-ranked

bookCount = defaultdict(int)
totalRead = 0

for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1

mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalRead/2: break

predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
    #header
        predictions.write(l)
        continue
    u,b = l.strip().split(',')
    if b in return1:
        predictions.write(u + ',' + b + ",1\n")
    else:
        predictions.write(u + ',' + b + ",0\n")

predictions.close()
#Load Data from Raw File
allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)
ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
booksPerUser = defaultdict(set)
usersPerBook = defaultdict(set)
booksInTrain=set()
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    booksPerUser[u].add(b)
    usersPerBook[b].add(u)
    booksInTrain.add(b)
### Would Read Prediction---------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------
# JaccardSimilarity Func
def JaccardSimilarity(set1,set2):
    inter=len(set1.intersection(set2))
    uni=len(set1.union(set2))
    return inter/uni

# define newreturn
newreturn1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    newreturn1.add(i)
    if count > 0.72*totalRead: break
        
#threshold
pct=0.02631578947368421### if use 'and': 0.001072961373390558  #
import csv
with open('predictions_Read.csv', 'w',newline="") as f:
    writer = csv.writer(f)
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            line=['userID','bookID','prediction']
            writer.writerow(line)
            continue
        u,b = l.strip().split(',')
        books=booksPerUser[u]
        users=usersPerBook[b]
        sim_max=0
        for comparedBook in books:
            sim_ij=JaccardSimilarity(users,usersPerBook[comparedBook])
            if sim_ij>sim_max:
                sim_max=sim_ij
        threshold=pct
        if sim_max>threshold or b in newreturn1:
            predict=1
        else:
            predict=0
        line=[u,b,str(predict)]
        writer.writerow(line)
### Rating Prediction-------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------
betaU = {}
betaI = {}
for u in ratingsPerUser:
    betaU[u] = 0

for b in ratingsPerItem:
    betaI[b] = 0

# MSE Function--------------------------------------------------------
def MSE(pred, label):
    err_sqr = [(x-y)**2 for x,y in zip(pred,label)]
    return sum(err_sqr) / len(err_sqr)

ratingMean =  sum([r for (_,_,r) in ratingsTrain]) / len(ratingsTrain)
alpha = ratingMean
def iterate(lamb):
    newAlpha = 0
    for u,b,r in ratingsTrain:
        newAlpha += r - (betaU[u] + betaI[b])
    alpha = newAlpha / len(ratingsTrain)
    for u in ratingsPerUser:
        newBetaU = 0
        for b,r in ratingsPerUser[u]:
            newBetaU += r - (alpha + betaI[b])
        betaU[u] = newBetaU / (lamb + len(ratingsPerUser[u]))
    for b in ratingsPerItem:
        newBetaI = 0
        for u,r in ratingsPerItem[b]:
            newBetaI += r - (alpha + betaU[u])
        betaI[b] = newBetaI / (lamb + len(ratingsPerItem[b]))
    mse = 0
    for u,b,r in ratingsTrain:
        prediction = alpha + betaU[u] + betaI[b]
        mse += (r - prediction)**2
    regularizer = 0
    for u in betaU:
        regularizer += betaU[u]**2
    for b in betaI:
        regularizer += betaI[b]**2
    mse /= len(ratingsTrain)
    return mse, mse + lamb*regularizer
lamb=4
mse,objective = iterate(lamb)
newMSE,newObjective = iterate(lamb)
iterations = 2
while iterations < 200 or objective - newObjective > 0.0001:
    mse, objective = newMSE, newObjective
    newMSE, newObjective = iterate(lamb)
    iterations += 1
    print("Objective after "
        + str(iterations) + " iterations = " + str(newObjective))
    print("MSE after "
        + str(iterations) + " iterations = " + str(newMSE))
# On ValidationSet--------------------------------------------------------
print("On ValidationSet")
specialCase=0
predValid=[]
alwaysPredictMean = [ratingMean for _ in ratingsValid]
labels = [r for _,_,r in ratingsValid]
specialCase=0
for (u,b,_) in ratingsValid:
    if u not in ratingsPerUser or b not in ratingsPerItem:
        predValid.append(ratingMean)
        specialCase+=1
    else:
        prediction=alpha + betaU[u] + betaI[b]
        predValid.append(prediction)
mseValid=MSE(predValid,labels)
print("MSE on ValidationSet:{}".format(mseValid))
print("Always predict Mean:{}".format(MSE(alwaysPredictMean, labels)))
print("Special Case Number:{}".format(specialCase))
# Predict and write--------------------------------------------------------
import csv
case=0
with open('predictions_Rating.csv', 'w',newline="") as f:
    writer = csv.writer(f)
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            line=['userID','bookID','prediction']
            writer.writerow(line)
            continue
        u,i = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
        if u not in ratingsPerUser or i not in ratingsPerItem:
            predict=ratingMean
            case+=1
        else:
            predict=alpha + betaU[u] + betaI[i]
        line=[u,i,str(predict)]
        writer.writerow(line)
    
print("Special Case:{}".format(case))