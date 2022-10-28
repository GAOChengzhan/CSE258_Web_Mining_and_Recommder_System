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
    d=dataTrain.iloc[i,:]
    user,book = d['user_id'], d['book_id']
    rating=d['rating']
    testRatingDict[(user,book)]=rating

#calculate global average ratings of all books
predictUserBook=[]
count=0
for (user,book) in testRatingDict:
    numerator=0
    denominator=0
    users= usersPerBook[book] 
    for comparedBook in booksPerUser[user]:
        if book == comparedBook: 
            continue
        sim_ij = JaccardSimilarity(users,usersPerBook[comparedBook])
        denominator+=(sim_ij)
        numerator+=((trainRatingDict[(user,comparedBook)]-avgRatingPerBook[comparedBook])*sim_ij)
    if  denominator!=0:
        ans=(avgRatingPerBook[book]+(numerator/denominator))  
    else:
        ans= globalAvg
        count+=1
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
    avgRating=avgRatingPerUser[user]
    numerator=0
    denominator=0
    books = booksPerUser[user] 
    for comparedUser in usersPerBook[book]:
        if user == comparedUser: 
            continue
        sim_uv = JaccardSimilarity(books,booksPerUser[comparedUser])
        denominator+=(sim_uv)
        numerator+=((trainRatingDict[(comparedUser,book)]-avgRatingPerUser[comparedUser])*sim_uv)
    ans=avgRating+ ((numerator/denominator) if  denominator!=0 else 0)
    if ans>5:
        ans=5
    if ans<0:
        ans=0
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
f.close()