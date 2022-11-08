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

### Category prediction baseline: Just consider some of the most common words from each category

catDict = {
  "children": 0,
  "comics_graphic": 1,
  "fantasy_paranormal": 2,
  "mystery_thriller_crime": 3,
  "young_adult": 4
}

predictions = open("predictions_Category.csv", 'w')
predictions.write("userID,reviewID,prediction\n")
for l in readGz("test_Category.json.gz"):
    cat = catDict['fantasy_paranormal'] # If there's no evidence, just choose the most common category in the dataset
    words = l['review_text'].lower()
    if 'children' in words:
        cat = catDict['children']
    if 'comic' in words:
        cat = catDict['comics_graphic']
    if 'fantasy' in words:
        cat = catDict['fantasy_paranormal']
    if 'mystery' in words:
        cat = catDict['mystery_thriller_crime']
    if 'love' in words:
        cat = catDict['young_adult']
    predictions.write(l['user_id'] + ',' + l['review_id'] + "," + str(cat) + "\n")

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
### Question 1--------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------
random.seed(33)#set random seed
addpairs=set()
addValid=[]
interactionValid=[]
for u,b,r in ratingsValid:
    #build
    interactionValid.append((u,b,1))
    #add
    readbooks=set(ratingsPerUser[u])
    unreadbooks=list(booksInTrain-readbooks)
    while True:
        number=random.randint(0, len(unreadbooks)-1)
        addbook=unreadbooks[number]
        if (u,addbook) not in addpairs:
            addValid.append((u,addbook,0))
            addpairs.add((u,addbook))
            break
interactionValid.extend(addValid)
print(len(interactionValid))
#calculate acc
TrueCount=0
for u,b,r in interactionValid:
    if b in return1 and r==1:
        TrueCount+=1
    if b not in return1 and r==0:
        TrueCount+=1
acc1=TrueCount/len(interactionValid)
print(acc1)
answers['Q1'] = acc1
assertFloat(answers['Q1'])
### Question 2--------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------
newreturn1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    newreturn1.add(i)
    if count > 3*totalRead/4: break
TrueCount=0
for u,b,r in interactionValid:
    if b in newreturn1 and r==1:
        TrueCount+=1
    if b not in newreturn1 and r==0:
        TrueCount+=1
acc2=TrueCount/len(interactionValid)
print(acc2)
answers['Q2'] = [3/4, acc2]
assertFloat(answers['Q2'][0])
assertFloat(answers['Q2'][1])
### Question 3--------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------
def JaccardSimilarity(set1,set2):
    inter=len(set1.intersection(set2))
    uni=len(set1.union(set2))
    return inter/uni

TrueCount=0
record=[]
for u,b,r in interactionValid:
    books=booksPerUser[u]
    users=usersPerBook[b]
    sim_max=0
    for comparedBook in books:
        sim_ij=JaccardSimilarity(users,usersPerBook[comparedBook])
        if sim_ij>sim_max:
            sim_max=sim_ij
    record.append(sim_max)
    threshold=0.0032679738562091504#48%
    if sim_max>threshold and r==1:
        TrueCount+=1
    if sim_max<threshold and r==0: 
        TrueCount+=1
acc3=TrueCount/len(interactionValid)
print(acc3)
### Question 4--------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------
newreturn1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    newreturn1.add(i)
    if count > 0.73*totalRead: break
import numpy as np
record=np.array(record)
pct=np.percentile(record,40)
print(pct)
TrueCount=0
for u,b,r in interactionValid:
    books=booksPerUser[u]
    users=usersPerBook[b]
    sim_max=0
    for comparedBook in books:
        sim_ij=JaccardSimilarity(users,usersPerBook[comparedBook])
        if sim_ij>sim_max:
            sim_max=sim_ij
    threshold=pct #0.0032679738562091504#48%
    if sim_max>threshold and b in newreturn1:
        if r==1:
            TrueCount+=1
    else:
        if r==0: 
            TrueCount+=1
acc4=TrueCount/len(interactionValid)
print(acc4)
answers['Q3'] = acc3
answers['Q4'] = acc4
assertFloat(answers['Q3'])
assertFloat(answers['Q4'])
### Question 5--------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------
# predictions = open("predictions_Read.csv", 'w')
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
        threshold=pct #0.000481000481000481
        if sim_max>threshold and b in newreturn1:
            predict=1
        else:
            predict=0
        line=[u,b,str(predict)]
        writer.writerow(line)

answers['Q5'] = "I confirm that I have uploaded an assignment submission to gradescope"
assert type(answers['Q5']) == str
### Question 9--------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------
import tensorflow as tf
userIDs,itemIDs = {},{}
for (u,i,_) in ratingsTrain:
#    u,i = d['user_id'],d['book_id']
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
)

class LatentFactorModelBiasOnly(tf.keras.Model):
    def __init__(self, mu, lamb):
        super(LatentFactorModelBiasOnly, self).__init__()
        # Initialize to average
        self.alpha = tf.Variable(mu)
        # Initialize to small random values
        self.betaU = tf.Variable(tf.random.normal([len(userIDs)],stddev=0.001))
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.lamb = lamb

    # Prediction for a single instance (useful for evaluation)
    def predict(self, u, i):
        p = self.alpha + self.betaU[u] + self.betaI[i]
        return p

    # L2 Regularizer
    def reg(self):
        return self.lamb * (tf.reduce_sum(self.betaU**2) +\
                            tf.reduce_sum(self.betaI**2))
    
    # Prediction for a sample of instances
    def predictSample(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_u = tf.nn.embedding_lookup(self.betaU, u)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        pred = self.alpha + beta_u + beta_i
        return pred
    
    # Loss
    def call(self, sampleU, sampleI, sampleR):
        pred = self.predictSample(sampleU, sampleI)
        r = tf.convert_to_tensor(sampleR, dtype=tf.float32)
        return tf.nn.l2_loss(pred - r) / len(sampleR)

mu = sum([r for _,_,r in ratingsTrain]) / len(ratingsTrain)
modelBiasOnly = LatentFactorModelBiasOnly(mu, 1)# lambda=1

def trainingStepBiasOnly(model, interactions):
    Nsamples = 5000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleR = [], [], []
        for _ in range(Nsamples):
            u,i,r = random.choice(interactions)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleR.append(r)

        loss = model(sampleU,sampleI,sampleR)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()

for i in range(150):
    obj = trainingStepBiasOnly(modelBiasOnly, ratingsTrain)
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))
#MSE Function
def MSE(pred, label):
    err_sqr = [(x-y)**2 for x,y in zip(pred,label)]
    return sum(err_sqr) / len(err_sqr)
alwaysPredictMean = [mu for _ in ratingsValid]
labels = [r for _,_,r in ratingsValid]
print(MSE(alwaysPredictMean, labels))
biasOnlyPredictions=[]
case=0
for u,i,_ in ratingsValid:
    if u not in userIDs or i not in itemIDs:
        biasOnlyPredictions.append(mu)
        case+=1
    else:
        biasOnlyPredictions.append(modelBiasOnly.predict(userIDs[u],itemIDs[i]).numpy())

validMSE=MSE(biasOnlyPredictions, labels)
print(validMSE)
print("Special Case:{}".format(case))
answers['Q9'] = validMSE
assertFloat(answers['Q9'])
### Question 10--------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------
betaU_lst=modelBiasOnly.betaU.numpy()
maxBeta,minBeta=0,0
for i,j in enumerate(betaU_lst):
    if j>maxBeta:
        maxBeta=j
        maxUser=i
    if j<minBeta:
        minBeta=j
        minUser=i
print([maxUser, minUser, maxBeta, minBeta])
userIDs,itemIDs = {},{}
for (u,i,_) in ratingsTrain:
#     u,i = d['user_id'],d['book_id']
    if not u in userIDs: 
        userIDs[u] = len(userIDs)
        if len(userIDs)==maxUser:
            maxUserNew=u
        if len(userIDs)==minUser:
            minUserNew=u
print([maxUserNew, minUserNew, maxBeta, minBeta])
answers['Q10'] = [maxUserNew, minUserNew, float(maxBeta), float(minBeta)]
assert [type(x) for x in answers['Q10']] == [str, str, float, float]
### Question 11--------------------------------------------------------------------------------------
### ------------------------------------------------------------------------------------------------
for Lambda in [1e-10,1e-8,1e-6,1e-5,1e-4,1e-3,0.01, 0.1, 10]: 
    userIDs,itemIDs = {},{}
    for (u,i,_) in ratingsTrain:
    #    u,i = d['user_id'],d['book_id']
        if not u in userIDs: userIDs[u] = len(userIDs)
        if not i in itemIDs: itemIDs[i] = len(itemIDs)
    modelBiasOnlylambda = LatentFactorModelBiasOnly(mu, Lambda)
    print("Start Training \nTrain Model when Lambda = {}".format(Lambda))
    for i in range(100):
        obj = trainingStepBiasOnly(modelBiasOnlylambda, ratingsTrain)
        if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))
    print("Training Finished")
    biasOnlyPredictions=[]
    for u,i,_ in ratingsValid:
        if u not in userIDs or i not in itemIDs:
            biasOnlyPredictions.append(mu)
        else:
            biasOnlyPredictions.append(modelBiasOnlylambda.predict(userIDs[u],itemIDs[i]).numpy())
    validMSELambda=MSE(biasOnlyPredictions, labels)
    print("MSE on ValidationSet:{}".format(validMSELambda))
(lamb, validMSE)=(1e-05,1.5920930827596471)
answers['Q11'] = (lamb, validMSE)
assertFloat(answers['Q11'][0])
assertFloat(answers['Q11'][1])

#On Test Data ---------------------------------------------------------------------------------------
Lambda=1e-5

userIDs,itemIDs = {},{}
for (u,i,_) in ratingsTrain:
#    u,i = d['user_id'],d['book_id']
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)
modelBiasOnlylambda = LatentFactorModelBiasOnly(mu, Lambda)
print("Start Training \nTrain Model when Lambda = {}".format(Lambda))
for i in range(100):
    obj = trainingStepBiasOnly(modelBiasOnlylambda, ratingsTrain)
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))
print("Training Finished")
biasOnlyPredictions=[]
for u,i,_ in ratingsValid:
    if u not in userIDs or i not in itemIDs:
        biasOnlyPredictions.append(mu)
    else:
        biasOnlyPredictions.append(modelBiasOnlylambda.predict(userIDs[u],itemIDs[i]).numpy())
validMSELambda=MSE(biasOnlyPredictions, labels)
print("MSE on ValidationSet:{}".format(validMSELambda))
case=0
userIDs,itemIDs = {},{}
for (u,i,_) in ratingsTrain:
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)
with open('predictions_Rating.csv', 'w',newline="") as f:
    writer = csv.writer(f)
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            line=['userID','bookID','prediction']
            writer.writerow(line)
            continue
        u,b = l.strip().split(',') # Read the user and item from the "pairs" file and write out your prediction
        if u not in userIDs or i not in itemIDs:
            predict=mu
            case+=1
        else:
            predict=modelBiasOnlylambda.predict(userIDs[u],itemIDs[i]).numpy()
        line=[u,b,str(predict)]
        writer.writerow(line)
    
print("Special Case:{}".format(case))
f = open("answers_hw3.txt", 'w')
f.write(str(answers) + '\n')
f.close()