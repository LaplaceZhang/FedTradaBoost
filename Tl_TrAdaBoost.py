# original: chenchiwei, edited by Laplace
# -*- coding: UTF-8 -*-
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn import decomposition

# import TrAdaBoost as tr
import Fed_GBDT as tr

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

import numpy as np


root_path = "I:/PytorchSave/local_save/data/IDS/"
DDoS = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
Bot = "Friday-WorkingHours-Morning.pcap_ISCX.csv"
Bot_test = "Friday-WorkingHours-Morning.pcap_ISCX.csv"
Port = 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'


def append_feature(dataframe, istest):
    lack_num = np.asarray(dataframe.isnull().sum(axis=1))
    # lack_num = np.asarray(dataframe..sum(axis=1))
    if istest:
        X = dataframe.values
        X = X[:, 1:X.shape[1]]
    else:
        X = dataframe.values
        X = X[:, 1:X.shape[1] - 1]
    total_S = np.sum(X, axis=1)
    var_S = np.var(X, axis=1)
    X = np.c_[X, total_S]
    X = np.c_[X, var_S]
    X = np.c_[X, lack_num]

    return X

# If use DataFrame, GPU computing gets disabled, considering no GPU for IoT, it is just fine.


train_df = pd.DataFrame(pd.read_csv(root_path + DDoS))
train_df.fillna(value=0, inplace=True)
train_df1 = pd.DataFrame(pd.read_csv(root_path + Bot))
train_df1.fillna(value=0, inplace=True)
test_df = pd.DataFrame(pd.read_csv(root_path + Bot_test))
train_df.fillna(value=0, inplace=True)

'''*********************
Re-allocate public samples based on weights
*********************'''
Wts = [1.0, 1.0, 1.0, 1.0]
Total_Sample = 30000
data_0 = train_df[train_df[" Label"]==0]
data_1 = train_df[train_df[" Label"]==1]
data_0b = train_df1[train_df1[" Label"]==0]
data_2 = train_df1[train_df1[" Label"]==2]
train_df = data_0.sample(n = int(Total_Sample * (Wts[0]/len(Wts))), replace=True).append(
    data_1.sample(n = int(Total_Sample * (Wts[1]/len(Wts))), replace=True))
train_df1 = data_0b.sample(n = int(Total_Sample * (Wts[3]/len(Wts))), replace=True).append(
    data_2.sample(n = int(Total_Sample * (Wts[2]/len(Wts))), replace=True))
test_data_S = data_0b.sample(n = int(Total_Sample * (Wts[3]/len(Wts))), replace=True).append(
    data_2.sample(n = int(Total_Sample * (Wts[2]/len(Wts))), replace=True))

train_data_T = train_df.values
train_data_S = train_df1.values
test_data_S = test_df.values

print('data loaded.')

# *************************
# Process data

label_T = train_data_T[:, train_data_T.shape[1] - 1]
# trans_T = train_data_T[:, 1:train_data_T.shape[1] - 1]
trans_T = append_feature(train_df, istest=False)

label_S = train_data_S[:, train_data_S.shape[1] - 1]
# trans_S = train_data_S[:, 1:train_data_S.shape[1] - 1]
trans_S = append_feature(train_df1, istest=False)

test_data_no = test_data_S[:, 0]
# test_data_S = test_data_S[:, 1:test_data_S.shape[1]]
test_data_S = append_feature(test_df, istest=True)

print('data split end.', trans_S.shape, trans_T.shape, label_S.shape, label_T.shape, test_data_S.shape)

# # 加上和、方差、缺失值数量的特征，效果有所提升
# trans_T = append_feature(trans_T, train_df)
# trans_S = append_feature(trans_S, train_df1)
# test_data_S = append_feature(test_data_S, test_df)
#
# print 'append feature end.', trans_S.shape, trans_T.shape, label_S.shape, label_T.shape, test_data_S.shape

# imputer_T = SimpleImputer(missing_values='nan', strategy='constant', fill_value=0)
# imputer_S = SimpleImputer(missing_values='nan', strategy='constant', fill_value=0)
# imputer_T.fit(trans_T,label_T)
# imputer_S.fit(trans_S, label_S)
#
# trans_T = imputer_S.transform(trans_T)
# trans_S = imputer_S.transform(trans_S)
#
# test_data_S = imputer_S.transform(test_data_S)

# pca_T = decomposition.PCA(n_components=50)
# pca_S = decomposition.PCA(n_components=50)
#
# trans_T = pca_T.fit_transform(trans_T)
# trans_S = pca_S.fit_transform(trans_S)
# test_data_S = pca_S.transform(test_data_S)

print('data preprocessed.', trans_S.shape, trans_T.shape, label_S.shape, label_T.shape, test_data_S.shape)
# *************************

X_train, X_test, y_train, y_test = model_selection.train_test_split(trans_S, label_S, test_size=0.33, random_state=42)

# feature scale
# scaler = preprocessing.StandardScaler()
# X_train = scaler.fit_transform(X_train, y_train)
# X_test = scaler.transform(X_test)
# print 'feature scaled end.'

pred, updated_model = tr.TrAdaBoost().tradaboost(X_train, trans_T, y_train, label_T, X_test, 3)
fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=pred, pos_label=0)
f = open('./local_save/model/model1.pickle', 'wb')
pickle.dump(updated_model, f)
f.close()
print(updated_model)
print('auc:', metrics.auc(fpr, tpr))
print("acc:", accuracy_score(pred, y_test))