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
# reviewsPerUser & reviewsPerItem
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
for (u,i,r) in ratingsTrain:
    reviewsPerUser[u].append((i,r))
    reviewsPerItem[i].append((u,r))

N = len(ratingsTrain)
nUsers = len(reviewsPerUser)
nItems = len(reviewsPerItem)
users = list(reviewsPerUser.keys())
items = list(reviewsPerItem.keys())

ratingMean =  sum([r for (_,_,r) in ratingsTrain]) / len(ratingsTrain)
alpha = ratingMean

userBiases = defaultdict(float)
itemBiases = defaultdict(float)

def prediction(user, item):
    return alpha + userBiases[user] + itemBiases[item]

#MSE Function
def MSE(pred, label):
    err_sqr = [(x-y)**2 for x,y in zip(pred,label)]
    return sum(err_sqr) / len(err_sqr)

def unpack(theta):
    global alpha
    global userBiases
    global itemBiases
    alpha = theta[0]
    userBiases = dict(zip(users, theta[1:nUsers+1]))
    itemBiases = dict(zip(items, theta[1+nUsers:]))
    
def cost(theta, labels, lamb):
    unpack(theta)
    predictions = [prediction(u,i) for (u,i,r) in ratingsTrain]
    cost = MSE(predictions, labels)
    print("MSE = " + str(cost))
    for u in userBiases:
        cost += lamb*userBiases[u]**2
    for i in itemBiases:
        cost += lamb*itemBiases[i]**2
    return cost

def derivative(theta, labels, lamb):
#     lamb=1e-5
    unpack(theta)
    N = len(ratingsTrain)
    dalpha = 0
    dUserBiases = defaultdict(float)
    dItemBiases = defaultdict(float)
    for (u,i,r) in ratingsTrain:
        pred = prediction(u, i)
        diff = pred - r
        dalpha += 2/N*diff
        dUserBiases[u] += 2/N*diff
        dItemBiases[i] += 2/N*diff
    for u in userBiases:
        dUserBiases[u] += 2*lamb*userBiases[u]
    for i in itemBiases:
        dItemBiases[i] += 2*lamb*itemBiases[i]
    dtheta = [dalpha] + [dUserBiases[u] for u in users] + [dItemBiases[i] for i in items]
    return numpy.array(dtheta)
#TrainingSet------------------------------------------------------------------------------------
print("On TrainingSet:")
labels = [r for (_,_,r) in ratingsTrain]
md=scipy.optimize.fmin_l_bfgs_b(cost, [alpha] + [0.0]*(nUsers+nItems),
                             derivative, args = (labels, 1e-5))
alwaysPredictMean = [ratingMean for _ in ratingsTrain]
labels = [r for _,_,r in ratingsTrain]
print("Always predict Mean:{}".format(MSE(alwaysPredictMean, labels)))
# ValidationSet------------------------------------------------------------------------------------
print("On ValidationSet")
specialCase=0
predValid=[]
alwaysPredictMean = [ratingMean for _ in ratingsValid]
labels = [r for _,_,r in ratingsValid]

for (u,i,r) in ratingsValid:
    if u not in userBiases or i not in itemBiases:
        predValid.append(ratingMean)
    else:
        predValid.append(prediction(u,i))
mseValid=MSE(predValid,labels)
print("MSE:{}".format(mseValid))
print("Always predict Mean:{}".format(MSE(alwaysPredictMean, labels)))
# TestSet------------------------------------------------------------------------------------
import csv
case=0
with open('predictions_Rating.csv', 'w',newline="") as f:
    writer = csv.writer(f)
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            line=['userID','bookID','prediction']
            writer.writerow(line)
            continue
        u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
        if u not in userBiases or i not in itemBiases:
            predict=ratingMean
            case+=1
        else:
            predict=prediction(u,i)
        line=[u,b,str(predict)]
        writer.writerow(line)
    
print("Special Case:{}".format(case))