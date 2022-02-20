# -*- coding: UTF-8 -*-
import pandas as pd
# from sklearn import preprocessing
# from sklearn import decomposition

# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # Adopt Random Forest (RF)
# from sklearn import svm
# from sklearn import feature_selection
from sklearn import model_selection
# from sklearn import metrics
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score

import numpy as np

root_path = "./local_save/client1/"
PublicData = 'Received.csv'
Bot = 'Bot.csv'  # 3000 normal + 2000 Bot
DDoS = 'DDoS.csv'  # 1000 normal + 4000 DDoS
Port = 'Port.csv'  # 2000 normal + 3000 Port
Bruf = 'Bruf.csv'  # 3000 normal + 2000 BruF
Infi = 'Inf.csv'  # 3000 normal + 2000 Infi
Test = 'Test_125.csv'  # 2500+1000+1000+1000+1000+1000 all six types, change num of class when create this dataset
LocalData = 'local1.csv'



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


train_target = pd.DataFrame(pd.read_csv(root_path + LocalData))  # Target
# Manually set a weak dataset for the target so we can show the advantages of transfer
train_target = train_target.sample(1000)
train_target.fillna(value=0, inplace=True)
train_source = pd.DataFrame(pd.read_csv(root_path + PublicData))  # Source
train_source.fillna(value=0, inplace=True)
test_df = pd.DataFrame(pd.read_csv(root_path + Test))  # test
test_df.fillna(value=0, inplace=True)

train_data_T = train_target.values
train_data_S = train_source.values
test_data = test_df.values

print('data loaded.')


# *************************
# Process data
def process():
    label_T = train_data_T[:, train_data_T.shape[1] - 1]
    trans_T = append_feature(train_target, istest=False)

    label_S = train_data_S[:, train_data_S.shape[1] - 1]
    trans_S = append_feature(train_source, istest=False)

    test_data_label = test_data[:, test_data.shape[1] - 1]
    test_data_S = append_feature(test_df, istest=False)

    print('data split end.', trans_S.shape, trans_T.shape, label_S.shape, label_T.shape, test_data_S.shape)

    # Source_data, Test_data, Source_label, Test_label = model_selection.train_test_split(trans_T, label_T, test_size=0.33,
    #                                                                     random_state=42)
    Source_data = trans_S
    Source_label = label_S
    Test_data = test_data_S
    Test_label = test_data_label
    Target_data = trans_T
    Target_label = label_T
    print(' Test data split end.', Source_data.shape, Test_data.shape, Source_label.shape, Test_label.shape)

    return Source_data, Test_data, Source_label, Test_label, Target_data, Target_label
