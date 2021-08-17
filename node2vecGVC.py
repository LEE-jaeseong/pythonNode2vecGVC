import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import minmax_scale
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from node2vec import Node2Vec
import math
import random
import gc

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

os.getcwd()
os.chdir('C://Users/owner/Documents/research/2021GVC/우리나라수출상위')

# 데이터 전처리
comtrade = pd.read_csv('Food_im_1_40_2020.csv',error_bad_lines=False, engine='python', encoding='CP949') #한글 깨짐 현상 해결
comtrade.dtypes #변수명 확인

comtrade = comtrade.loc[:,['Reporter','Partner','Trade Value (US$)']]

def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
        {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)

comtrade.shape
comtrade = comtrade.loc[which(comtrade.loc[:,'Partner']!='World'),:]

tradeValueScaled = minmax_scale(comtrade.loc[:,'Trade Value (US$)'])
comtrade.loc[:,'Trade Value (US$)'] = tradeValueScaled
comtrade.head()

# 네트워크 변환
G = nx.Graph()
G = nx.from_pandas_edgelist(comtrade, 'Reporter','Partner', create_using=nx.Graph())

leaderboard = {}
for x in G.nodes:
 leaderboard[x] = len(G[x])
s = pd.Series(leaderboard, name='connections')
df2 = s.to_frame().sort_values('connections', ascending=False)
df2.head()

# plot graph
plt.figure(figsize=(10,10))

pos = nx.random_layout(G, seed=23)
nx.draw(G, with_labels=False,  pos = pos, node_size = 40, alpha = 0.6, width = 0.7)

plt.show()

"""
# Retrieve Unconnected Node Pairs - Negative Samples
####################################################

# remove duplicate items from the list
nodeList = list(dict.fromkeys(comtrade.iloc[:,0]))

# build adjacency matrix
adj_G = nx.to_numpy_matrix(G, nodelist=nodeList)
adj_G.shape

# get unconnected node-pairs
all_unconnected_pairs = []

# traverse adjacency matrix
offset = 0
for i in tqdm(range(adj_G.shape[0])):
    for j in range(offset,adj_G.shape[1]):
        if i != j:
#            if nx.shortest_path_length(G, str(i), str(j)) <= 2:
            if adj_G[i,j] == 0:
                all_unconnected_pairs.append([nodeList[i],nodeList[j]])
    offset = offset + 1

all_unconnected_pairs
len(all_unconnected_pairs)

Reporter_unlinked = [i[0] for i in all_unconnected_pairs]
Partner_unlinked = [i[1] for i in all_unconnected_pairs]

data = pd.DataFrame({'Reporter':Reporter_unlinked,
                     'Partner': Partner_unlinked})
data['link'] = 0
print(data)

# Remove Links from Connected Node Pairs - Positive Samples
####################################################

initial_node_count = len(G.nodes)

comtrade_temp = comtrade.copy()

# empty list to store removable links
omissible_links_index = []

for i in tqdm(comtrade.index.values):

    # remove a node pair and build a new graph
    G_temp = nx.from_pandas_edgelist(comtrade.drop(index=i), "Reporter", "Partner", create_using=nx.Graph())

    # check there is no spliting of graph and number of nodes is same
    if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
        omissible_links_index.append(i)
        comtrade_temp = comtrade_temp.drop(index=i)

len(omissible_links_index)

# create dataframe of removable edges
comtrade_ghost = comtrade.loc[omissible_links_index]

# add the target variable 'link'
comtrade_ghost['link'] = 1

data = data.append(comtrade_ghost[['Reporter', 'Partner', 'link']], ignore_index=True)
data['link'].value_counts()
print(data)

## drop removable edges
comtrade_partial = comtrade.drop(index=comtrade_ghost.index.values)

## build graph
G_data = nx.from_pandas_edgelist(comtrade_partial, "Reporter", "Partner", create_using=nx.Graph())
print(len(G_data))

# Building our Link Prediction Model
####################################################

xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'],
                                                test_size = 0.3,
                                                random_state = 35)

lr = LogisticRegression(class_weight="balanced")
lr.fit(xtrain, ytrain)

predictions = lr.predict_proba(xtest)
print(roc_auc_score(ytest, predictions[:,1]))
gc.collect()



import lightgbm as lgbm

train_data = lgbm.Dataset(xtrain, ytrain)
test_data = lgbm.Dataset(xtest, ytest)

# define parameters
parameters = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'num_threads' : 2,
    'seed' : 76
}

# train lightGBM model
model = lgbm.train(parameters,
                   train_data,
                   valid_sets=test_data,
                   num_boost_round=1000,
                   early_stopping_rounds=20)
"""

# JSLEE version of drop removabal edges
G_data = nx.from_pandas_edgelist(comtrade, "Reporter", "Partner", create_using=nx.Graph())
print(len(G_data))

# Generate walks
node2vec = Node2Vec(G_data, dimensions=100, walk_length=16, num_walks=50)

# train node2vec model
n2w_model = node2vec.fit(window=7, min_count=1)

x = [(n2w_model[str(i)]+n2w_model[str(j)]) for i,j in zip(comtrade['Reporter'], comtrade['Partner'])]

countryCode = []
for num in range(len(list(n2w_model.wv.vocab))):
    countryCode.append([num, list(n2w_model.wv.vocab)[num]])

countryCode[102]





comtrade['link'] = 1
tradeMean = comtrade['Trade Value (US$)'].describe()[1]
tradeMedian = comtrade['Trade Value (US$)'].describe()[5]
tradeQ1 = comtrade['Trade Value (US$)'].describe()[6]

random.seed(1)
trainIdx = random.sample(range(len(comtrade)), math.floor(len(x)*0.7))
valIdx = list(set(range(len(comtrade))).difference(trainIdx))

def cut_off(y, threshold) :
    Y = y.copy()  # 대문자 Y를 새로운 변수로 하여 기존의 y값에 영향이 가지 않도록 한다.
    Y[Y>threshold] = 1
    Y[Y<threshold] = 0
    return Y.astype(int)

#########
# 전체
#########

train_temp = comtrade.iloc[trainIdx,] # 70%
val_temp = comtrade.iloc[valIdx,] # 30%

random.seed(1)
train_tempIdx = random.sample(range(len(train_temp)), math.floor(len(train_temp)*0.2))

random.seed(1)
val_tempIdx = random.sample(range(len(val_temp)), math.floor(len(val_temp)*0.2))

train_temp.iloc[train_tempIdx,3] = 0
train_temp.iloc[:,3].value_counts()
#train_temp.to_csv('전체train.csv', index = False)

val_temp.iloc[val_tempIdx,3] = 0
val_temp.iloc[:,3].value_counts()
#val_temp.to_csv('전체test.csv', index = False)

xtrain = np.array(x)[trainIdx]
xtest = np.array(x)[valIdx]
ytrain = train_temp.iloc[:,3]
ytest = val_temp.iloc[:,3]

lr = LogisticRegression(class_weight="balanced")
lr.fit(xtrain, ytrain)

predictions = lr.predict_proba(xtest)
print(roc_auc_score(ytest, predictions[:,1]))

threshold = 0.65
print(precision_score(ytest, cut_off(predictions[:,1], threshold)))
print(recall_score(ytest, cut_off(predictions[:,1], threshold)))
print(f1_score(ytest, cut_off(predictions[:,1], threshold)))

print(round(pd.DataFrame(cut_off(predictions[:,1], threshold)).value_counts()[1] / len(val_temp.iloc[:,3]), 2)) # test 성능

# 추가 모델 학습
import lightgbm as lgbm

train_data = lgbm.Dataset(xtrain, ytrain)
test_data = lgbm.Dataset(xtest, ytest)

# define parameters
parameters = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'num_threads' : 2,
    'seed' : 76
}


# def lgb_f1_score(y_hat, data):
#     y_true = data.get_label()
#     y_hat = np.where(y_hat >= 0.5, 1, 0)   # scikits f1 doesn't like probabilities
#     return 'f1', f1_score(y_true, y_hat), True # f1_score, precision_score, recall_score


# train lightGBM model
model = lgbm.train(parameters,
                   train_data,
                   #feval = lgb_f1_score,
                   valid_sets=test_data,
                   num_boost_round=1000,
                   early_stopping_rounds=20)

predictions = model.predict(xtest)
print(roc_auc_score(ytest, predictions))

model = lgbm.train(parameters,
                   train_data,
                   valid_sets=test_data,
                   num_boost_round=4,
                   early_stopping_rounds=20)

threshold = 0.9
print(precision_score(ytest, cut_off(predictions, threshold)))
print(recall_score(ytest, cut_off(predictions, threshold)))
print(f1_score(ytest, cut_off(predictions, threshold)))

print(round(pd.DataFrame(cut_off(predictions, threshold)).value_counts()[1] / len(val_temp.iloc[:,3]), 2)) # test 성능





#########
# 평균 이상
#########

train_temp = comtrade.iloc[trainIdx,] # 70%
val_temp = comtrade.iloc[valIdx,] # 30%

valMeanTrain = set(which(train_temp['link'] == 1)) & set(which(train_temp['Trade Value (US$)'] >= tradeMean))
valMeanVal = set(which(val_temp['link'] == 1)) & set(which(val_temp['Trade Value (US$)'] >= tradeMean))

train_temp.iloc[list(valMeanTrain),3] = 0
train_temp['link'].value_counts()
#train_temp.to_csv('평균이상train.csv', index = False)

val_temp.iloc[list(valMeanVal),3] = 0
val_temp['link'].value_counts()
#val_temp.to_csv('평균이상test.csv', index = False)

xtrain = np.array(x)[trainIdx]
xtest = np.array(x)[valIdx]
ytrain = train_temp.iloc[:,3]
ytest = val_temp.iloc[:,3]

lr = LogisticRegression(class_weight="balanced")
lr.fit(xtrain, ytrain)

predictions = lr.predict_proba(xtest)
print(roc_auc_score(ytest, predictions[:,1]))

threshold = 0.5
print(precision_score(ytest, cut_off(predictions[:,1], threshold)))
print(recall_score(ytest, cut_off(predictions[:,1], threshold)))
print(f1_score(ytest, cut_off(predictions[:,1], threshold)))

print(round(pd.DataFrame(cut_off(predictions[:,1], threshold)).value_counts()[1] / len(val_temp.iloc[:,3]), 2)) # test 성능

# 추가 모델 학습
import lightgbm as lgbm

train_data = lgbm.Dataset(xtrain, ytrain)
test_data = lgbm.Dataset(xtest, ytest)

# define parameters
parameters = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'num_threads' : 2,
    'seed' : 76
}


# def lgb_f1_score(y_hat, data):
#     y_true = data.get_label()
#     y_hat = np.where(y_hat >= 0.5, 1, 0)   # scikits f1 doesn't like probabilities
#     return 'f1', f1_score(y_true, y_hat), True # f1_score, precision_score, recall_score


# train lightGBM model
model = lgbm.train(parameters,
                   train_data,
                   #feval = lgb_f1_score,
                   valid_sets=test_data,
                   num_boost_round=1000,
                   early_stopping_rounds=20)

predictions = model.predict(xtest)
print(roc_auc_score(ytest, predictions))

model = lgbm.train(parameters,
                   train_data,
                   valid_sets=test_data,
                   num_boost_round=46,
                   early_stopping_rounds=20)

threshold = 0.5
print(precision_score(ytest, cut_off(predictions, threshold)))
print(recall_score(ytest, cut_off(predictions, threshold)))
print(f1_score(ytest, cut_off(predictions, threshold)))

print(round(pd.DataFrame(cut_off(predictions, threshold)).value_counts()[1] / len(val_temp.iloc[:,3]), 2)) # test 성능



koreaPrediction = []
for country in range(len(countryCode)):
    koreaPrediction.append([countryCode[country][1],model.predict(n2w_model.wv.vectors[[102, country]])[1]])

koreaPrediction = pd.DataFrame(koreaPrediction)
koreaPrediction.to_csv('한국수출잠재예측국가.csv', index = False)

koreaPredictionThreshold = 0.7
for partner in which(koreaPrediction[1] < koreaPredictionThreshold):
    print(countryCode[partner][1])




########
# Q1 이상
########

train_temp = comtrade.iloc[trainIdx,] # 70%
val_temp = comtrade.iloc[valIdx,] # 30%

valQ1Train = set(which(train_temp['link'] == 1)) & set(which(train_temp['Trade Value (US$)'] >= tradeQ1))
valQ1Val = set(which(val_temp['link'] == 1)) & set(which(val_temp['Trade Value (US$)'] >= tradeQ1))

train_temp.iloc[list(valQ1Train),3] = 0
train_temp['link'].value_counts()
#train_temp.to_csv('Q1이상train.csv', index = False)

val_temp.iloc[list(valQ1Val),3] = 0
val_temp['link'].value_counts()
#val_temp.to_csv('Q1이상test.csv', index = False)

xtrain = np.array(x)[trainIdx]
xtest = np.array(x)[valIdx]
ytrain = train_temp.iloc[:,3]
ytest = val_temp.iloc[:,3]

lr = LogisticRegression(class_weight="balanced")
lr.fit(xtrain, ytrain)

predictions = lr.predict_proba(xtest)
print(roc_auc_score(ytest, predictions[:,1]))

threshold = 0.9
print(precision_score(ytest, cut_off(predictions[:,1], threshold)))
print(recall_score(ytest, cut_off(predictions[:,1], threshold)))
print(f1_score(ytest, cut_off(predictions[:,1], threshold)))

print(round(pd.DataFrame(cut_off(predictions[:,1], threshold)).value_counts()[1] / len(val_temp.iloc[:,3]), 2)) # test 성능


# 추가 모델 학습
import lightgbm as lgbm

train_data = lgbm.Dataset(xtrain, ytrain)
test_data = lgbm.Dataset(xtest, ytest)

# define parameters
parameters = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'num_threads' : 2,
    'seed' : 76
}


# def lgb_f1_score(y_hat, data):
#     y_true = data.get_label()
#     y_hat = np.where(y_hat >= 0.5, 1, 0)   # scikits f1 doesn't like probabilities
#     return 'f1', f1_score(y_true, y_hat), True # f1_score, precision_score, recall_score


# train lightGBM model
model = lgbm.train(parameters,
                   train_data,
                   #feval = lgb_f1_score,
                   valid_sets=test_data,
                   num_boost_round=1000,
                   early_stopping_rounds=20)

predictions = model.predict(xtest)
print(roc_auc_score(ytest, predictions))

model = lgbm.train(parameters,
                   train_data,
                   valid_sets=test_data,
                   num_boost_round=46,
                   early_stopping_rounds=20)

threshold = 0.9
print(precision_score(ytest, cut_off(predictions, threshold)))
print(recall_score(ytest, cut_off(predictions, threshold)))
print(f1_score(ytest, cut_off(predictions, threshold)))

print(round(pd.DataFrame(cut_off(predictions, threshold)).value_counts()[1] / len(val_temp.iloc[:,3]), 2)) # test 성능




