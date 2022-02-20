# -*- coding: UTF-8 -*-
import socket
from threading import Thread
import pickle
import Fed_main as tr
import pandas as pd
import numpy as np
import random
import os
import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier  # Adopt Random Forest (RF)
from sklearn import model_selection

# from sklearn.tree import DecisionTreeClassifier
# from sklearn import svm
# from sklearn import feature_selection

"""
This function aims to implement federated average via FedAvg

Re-allocate public samples based on weights
Under one assumption: num of base_estimator is same in m1 & m2
"""


def WeightedSample(path, all_in_one, data_to_send, wts, N):
    """
    This function aims to use weight feedback to adjust public dataset so that it can fit the demand of clients.
    We assume each client has same contribution weights.
    Return: pd.DataFrame
    """

    # Read all instances
    train_data = pd.DataFrame(pd.read_csv(path + all_in_one))
    train_data.fillna(value=0, inplace=True)
    data_normal = train_data[train_data["Label"] == 0]
    data_ddos = train_data[train_data["Label"] == 1]
    data_bot = train_data[train_data["Label"] == 2]
    data_port = train_data[train_data["Label"] == 3]

    f = open(path + data_to_send, "rb")
    old_send = pickle.load(f)
    f.close()
    weights = np.load(path + wts, allow_pickle=True)  # np.array
    weights = weights.mean(axis=0)  # mean returned weights from client1 & 2 for they share same public samples
    old_normal = old_send[old_send["Label"] == 0]
    wts_normal = weights[0:len(old_normal)]
    old_ddos = old_send[old_send["Label"] == 1]
    wts_ddos = weights[len(old_normal):len(old_normal) + len(old_ddos)]
    old_bot = old_send[old_send["Label"] == 2]
    wts_bot = weights[len(old_normal) + len(old_ddos):len(old_normal) + len(old_ddos) + len(old_bot)]
    old_port = old_send[old_send["Label"] == 3]
    wts_port = weights[-len(old_port):]
    mid_normal = np.mean(wts_normal)  # Here we can set different weights or model iteration strategy
    mid_ddos = np.mean(wts_ddos)
    mid_bot = np.mean(wts_bot)
    mid_port = np.mean(wts_port)

    keep = old_send.reset_index(drop=True)
    for i in range(N):
        if old_send.iat[i, -1] == 0.0:
            if weights[i] > mid_normal:  # here we can use other methods to say an instance is good, not by median.
                continue
            else:
                keep.drop(labels=[i], axis=0, inplace=True)
        elif old_send.iat[i, -1] == 1.0:
            if weights[i] > mid_ddos:
                continue
            else:
                keep.drop(labels=[i], axis=0, inplace=True)
        elif old_send.iat[i, -1] == 2.0:
            if weights[i] > mid_bot:
                continue
            else:
                keep.drop(labels=[i], axis=0, inplace=True)
        else:
            if weights[i] > mid_port:
                continue
            else:
                keep.drop(labels=[i], axis=0, inplace=True)

    fix = N - len(keep)
    num_ddos = int(fix * mid_ddos / (mid_ddos + mid_bot + mid_port + mid_normal))
    num_bot = int(fix * mid_bot / (mid_ddos + mid_bot + mid_port + mid_normal))
    num_port = int(fix * mid_port / (mid_ddos + mid_bot + mid_port + mid_normal))
    num_normal = int(fix - (num_ddos + num_bot + num_port))
    name = [num_normal, num_ddos, num_bot, num_port]  # assign class distribution
    public_sample = keep.append(data_normal.sample(n=name[0], replace=True))
    public_sample = public_sample.append(data_ddos.sample(n=name[1], replace=True))
    public_sample = public_sample.append(data_bot.sample(n=name[2], replace=True))
    public_sample = public_sample.append(data_port.sample(n=name[3], replace=True))
    public_sample = public_sample.sort_values(by="Label", inplace=True)

    return public_sample


def Initialization(path, all_in_one, N):
    train_data = pd.DataFrame(pd.read_csv(path + all_in_one))
    train_data.fillna(value=0, inplace=True)
    data_normal = train_data[train_data["Label"] == 0]
    data_ddos = train_data[train_data["Label"] == 1]
    data_bot = train_data[train_data["Label"] == 2]
    data_port = train_data[train_data["Label"] == 3]
    data_bruf = train_data[train_data["Label"] == 4]
    data_inf = train_data[train_data["Label"] == 5]
    wts = np.ones([2, N]) / N  # equal class types
    np.save('{path}weights.npy'.format(path=path), wts)
    # assign public dataset
    public_sample = data_normal.sample(n=int(N * 4 / 20), replace=True)
    public_sample = public_sample.append(data_ddos.sample(n=int(N * 4 / 20), replace=True))
    public_sample = public_sample.append(data_bot.sample(n=int(N * 4 / 20), replace=True))
    public_sample = public_sample.append(data_port.sample(n=int(N * 4 / 20), replace=True))
    public_sample = public_sample.append(data_bruf.sample(n=int(N * 2 / 20), replace=True))
    public_sample = public_sample.append(data_inf.sample(n=int(N * 2 / 20), replace=True))
    public_sample.fillna(0, inplace=True)

    public_sample.sort_values(by='Label', inplace=True)  # ESSENTIAL FOR MODEL AGGREGATION!
    # return type: pd.DataFrame
    return public_sample


def adaboost(X, Y, m1, m2):
    UpdatedBaseEstr = []
    UpdatedBaseWeit = m1.estimator_weights_
    model = m1
    for i in range(len(m1.estimators_)):
        if m1.estimators_[i].score(X, Y) > m2.estimators_[i].score(X, Y):
            UpdatedBaseEstr.append(m1.estimators_[i])
            UpdatedBaseWeit[i] = m1.estimator_weights_[i]
        else:
            UpdatedBaseEstr.append(m2.estimators_[i])
            UpdatedBaseWeit[i] = m2.estimator_weights_[i]
    model.estimators_ = UpdatedBaseEstr
    model.estimator_weights_ = UpdatedBaseWeit

    return model


def FedBoost(wts, X, Y, m1, m2):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)  # RF exp

    model = clf.fit(X, Y)
    est_wts1 = m1.estimator_weights_
    est_wts2 = m2.estimator_weights_
    est_wts = np.append(est_wts1, est_wts2)
    UpdatedBaseEst = []
    UpdatedBaseWet = np.zeros(est_wts1.shape[0], )
    UpdatedErr = np.zeros(est_wts1.shape[0], )
    score = np.zeros(est_wts.shape[0])
    Num_0 = Y[Y == 0].shape[0]
    Num_1 = Y[Y == 1].shape[0]
    Num_2 = Y[Y == 2].shape[0]
    Num_3 = Y[Y == 3].shape[0]
    Num_4 = Y[Y == 4].shape[0]
    Num_5 = Y[Y == 5].shape[0]
    wts1_0 = wts.sum(wts[:Num_0, ], axis=1)
    wts1_1 = wts.sum(wts[Num_0:Num_0 + Num_1, ], axis=1)
    wts1_2 = wts.sum(wts[Num_0 + Num_1:Num_0 + Num_1 + Num_2, ], axis=1)
    wts1_3 = wts.sum(wts[Num_0 + Num_1 + Num_2:Num_0 + Num_1 + Num_2 + Num_3, ], axis=1)
    wts1_4 = wts.sum(wts[Num_0 + Num_1 + Num_2 + Num_3:Num_0 + Num_1 + Num_2 + Num_3 + Num_4, ], axis=1)
    wts1_5 = wts.sum(wts[-Num_5:, ], axis=1)
    wts1_sum = wts1_0 + wts1_1 + wts1_2 + wts1_3 + wts1_4 + wts1_5
    for i in range(est_wts1.shape[0]):
        score[i] = m1.estimators_[i].score(X, Y)
    for i in range(est_wts2.shape[0]):
        score[m1.estimator_weights_.shape[0] + i] = m2.estimators_[i].score(X, Y)
    # The range here need to be considered carefully to avoid over-fit. Try from 30% (after multiple attempts)
    rank = np.argsort(score)[::-1][175:est_wts1.shape[0] + 175]
    for i in range(rank.shape[0]):
        if rank[i] < est_wts1.shape[0]:
            UpdatedBaseEst.append(m1.estimators_[rank[i]])
            UpdatedBaseWet[i] = m1.estimator_weights_[rank[i]]
            UpdatedErr[i] = m1.estimator_errors_[rank[i]]
        else:
            UpdatedBaseEst.append(m2.estimators_[rank[i] - est_wts1.shape[0]])
            UpdatedBaseWet[i] = m2.estimator_weights_[rank[i] - est_wts1.shape[0]]
            UpdatedErr[i] = m2.estimator_errors_[rank[i] - est_wts1.shape[0]]

    model.estimators_ = UpdatedBaseEst
    model.estimator_weights_ = UpdatedBaseWet
    model.estimator_errors_ = UpdatedErr

    return model


# load pickle file & csv, DF only
def DataFrame_loader(path):
    filetype = (path).split('.')[-1]
    if filetype == 'pickle':
        f = open(path, "rb")
        df = pickle.load(f)
        f.close()
    elif filetype == 'csv':
        df = pd.read_csv(path)
    else:
        print("File type not allowed!")
    data = df.values
    Y = data[:, data.shape[1] - 1]
    # trans_T = train_data_T[:, 1:train_data_T.shape[1] - 1]
    X = append_feature(df, istest=False)
    return X, Y


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


def FedRank(X, Y, m1, m2, m3, m4, m5):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)  # RF exp

    model = clf.fit(X, Y)
    wts1 = m1.estimator_weights_
    wts2 = m2.estimator_weights_
    wts3 = m3.estimator_weights_
    wts4 = m4.estimator_weights_
    wts5 = m5.estimator_weights_
    est_wts = np.concatenate((wts1, wts2, wts3, wts4, wts5), axis=0)
    N = wts1.shape[0]
    UpdatedBaseEst = []
    UpdatedBaseWet = np.zeros(N, )
    UpdatedErr = np.zeros(N, )
    score = np.zeros(est_wts.shape[0])
    for i in range(wts1.shape[0]):
        score[i] = m1.estimators_[i].score(X, Y)
    for i in range(wts2.shape[0]):
        score[N + i] = m2.estimators_[i].score(X, Y)
    for i in range(wts3.shape[0]):
        score[2 * N + i] = m3.estimators_[i].score(X, Y)
    for i in range(wts4.shape[0]):
        score[3 * N + i] = m4.estimators_[i].score(X, Y)
    for i in range(wts5.shape[0]):
        score[4 * N + i] = m5.estimators_[i].score(X, Y)
    # The range here need to be considered carefully to avoid over-fit. Try from 30% (after multiple attempts)
    rank = np.argsort(score)[::-1][150:N + 150]
    for i in range(rank.shape[0]):
        if rank[i] in range(0, N):
            UpdatedBaseEst.append(m1.estimators_[rank[i]])
            UpdatedBaseWet[i] = m1.estimator_weights_[rank[i]]
            UpdatedErr[i] = m1.estimator_errors_[rank[i]]
        elif rank[i] in range(N, 2 * N):
            UpdatedBaseEst.append(m2.estimators_[rank[i] - N])
            UpdatedBaseWet[i] = m2.estimator_weights_[rank[i] - N]
            UpdatedErr[i] = m2.estimator_errors_[rank[i] - N]
        elif rank[i] in range(2 * N, 3 * N):
            UpdatedBaseEst.append(m3.estimators_[rank[i] - 2 * N])
            UpdatedBaseWet[i] = m3.estimator_weights_[rank[i] - 2 * N]
            UpdatedErr[i] = m3.estimator_errors_[rank[i] - 2 * N]
        elif rank[i] in range(3 * N, 4 * N):
            UpdatedBaseEst.append(m4.estimators_[rank[i] - 3 * N])
            UpdatedBaseWet[i] = m4.estimator_weights_[rank[i] - 3 * N]
            UpdatedErr[i] = m4.estimator_errors_[rank[i] - 3 * N]
        else:
            UpdatedBaseEst.append(m5.estimators_[rank[i] - 4 * N])
            UpdatedBaseWet[i] = m5.estimator_weights_[rank[i] - 4 * N]
            UpdatedErr[i] = m5.estimator_errors_[rank[i] - 4 * N]

    model.estimators_ = UpdatedBaseEst
    model.estimator_weights_ = UpdatedBaseWet
    model.estimator_errors_ = UpdatedErr
    print(model)

    return model


def FedAggDT(m1, m2, m3, m4, m5):
    """
    This module aim to implement the aggregation algorithm: Federated Averaging.
    To make this module work, we average all parameter from DT which we can access in the sklearn-adaboost models.
    NOTE: you need to check the details about all parameters to do average!
        The DT as base estimator only require average the random_state which involved in sklearn-DT
    :param m1: model from client 1
    :param m2: model from client 2
    :param m3: model from client 3
    :param m4: model from client 4
    :param m5: model from client 5
    :return: aggregated model m0 as for the public model in new iteration of training (OR output without )

    Here I list some parameters for AdaBoost in scikit-learn which you may need to do average:
    clf.estimator_weights_ --> ndarray: (500,)
    clf.estimator_errors_ --> ndarray: (500,)
    clf.estimators_ --> list: 500
    clf.feature_importances_ --> ndarray: (79,)
    clf.classes --> ndarray: (6,)
    As for the parameters in DecisionTree, only need to average the random_state: clf.estimators_[n].random_state
    Federated Averaging aims to do average for all unequal parameters in the updated models.
    In AdaBoost+DT, the above parameters are enough for FedAvg.
    """
    clf = m1
    # for i in range(clf.estimator_weights_.shape[0]):
    #     clf.estimator_weights_[i] = np.mean([m1.estimator_weights_[i], m2.estimator_weights_[i],
    #                                          m3.estimator_weights_[i], m4.estimator_weights_[i],
    #                                          m5.estimator_weights_[i]])
    # for i in range(clf.estimator_errors_.shape[0]):
    #     clf.estimator_errors_[i] = np.mean([m1.estimator_errors_[i], m2.estimator_errors_[i],
    #                                         m3.estimator_errors_[i], m4.estimator_errors_[i],
    #                                         m5.estimator_errors_[i]])
    # for i in range(clf.feature_importances_.shape[0]):
    #     clf.feature_importances_[i] = np.mean([m1.feature_importances_[i], m2.feature_importances_[i],
    #                                            m3.feature_importances_[i], m4.feature_importances_[i],
    #                                            m5.feature_importances_[i]])
    for i in range(len(clf.estimators_)):
        clf.estimators_[i].random_state = np.mean([m1.estimators_[i].random_state, m2.estimators_[i].random_state,
                                                   m3.estimators_[i].random_state, m4.estimators_[i].random_state,
                                                   m5.estimators_[i].random_state])

    return clf


def customize(wts, X, Y, public, client):
    """
    This function runs after aggregation is done. Aims to implement model personalization for each client and output
    accuracy score.
    Main steps:
    1. rank score with weight returned from clients for each estimator in aggregated model & client model (1000)
    2. choose top 500 for a new model (may not from the beginning, try Rank 150~650)

    :param wts: weight array for client0
    :param public: aggregated model
    :param client: updated model from client1
    :param X: public data
    :param Y: public data labels
    :return: updated model
    """
    m_new = client
    eswts = client.estimator_weights_
    eswts_pub = public.estimator_weights_
    UpdatedBaseEst = []
    UpdatedBaseWet = np.zeros(eswts.shape[0], )
    UpdatedErr = np.zeros(eswts.shape[0], )
    score = np.zeros(eswts.shape[0])
    score_pub = np.zeros(eswts_pub.shape[0])
    for i in range(eswts.shape[0]):
        score[i] = client.estimators_[i].score(X, Y, sample_weight=wts / np.sum(wts))
    rank = np.argsort(score)[::-1][150:(eswts.shape[0] * 1 // 5) + 150]
    for i in range(eswts.shape[0]):
        score_pub[i] = public.estimators_[i].score(X, Y, sample_weight=wts / np.sum(wts))
    rank_pub = np.argsort(score_pub)[::-1][150:(eswts.shape[0] * 1 // 4) + 150]

    for i in range(rank.shape[0]):
        UpdatedBaseEst.append(public.estimators_[rank[i]])
        UpdatedBaseWet[i] = public.estimator_weights_[rank[i]]
        UpdatedErr[i] = public.estimator_errors_[rank[i]]
    for i in range(rank_pub.shape[0]):
        UpdatedBaseEst.append(client.estimators_[rank_pub[i]])
        UpdatedBaseWet[i] = client.estimator_weights_[rank_pub[i]]
        UpdatedErr[i] = client.estimator_errors_[rank_pub[i]]

    m_new.estimators_ = UpdatedBaseEst
    m_new.estimator_weights_ = UpdatedBaseWet
    m_new.estimator_errors_ = UpdatedErr
    m_new.n_estimators = eswts.shape[0] * 1 // 5 + eswts.shape[0] * 1 // 4
    return m_new


def PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, clf):
    # *****************************************
    # Return acc for all single type
    df = pd.read_csv(data_path + testddos)
    data = df.values
    tY = data[:, data.shape[1] - 1]
    # trans_T = train_data_T[:, 1:train_data_T.shape[1] - 1]
    tX = append_feature(df, istest=False)
    print("DDoS score: {score}".format(score=clf.score(tX, tY)))
    df = pd.read_csv(data_path + testbot)
    data = df.values
    tY = data[:, data.shape[1] - 1]
    # trans_T = train_data_T[:, 1:train_data_T.shape[1] - 1]
    tX = append_feature(df, istest=False)
    print("Bot score: {score}".format(score=clf.score(tX, tY)))
    df = pd.read_csv(data_path + testport)
    data = df.values
    tY = data[:, data.shape[1] - 1]
    # trans_T = train_data_T[:, 1:train_data_T.shape[1] - 1]
    tX = append_feature(df, istest=False)
    print("Port score: {score}".format(score=clf.score(tX, tY)))
    df = pd.read_csv(data_path + testbruf)
    data = df.values
    tY = data[:, data.shape[1] - 1]
    # trans_T = train_data_T[:, 1:train_data_T.shape[1] - 1]
    tX = append_feature(df, istest=False)
    print("BruteForce score: {score}".format(score=clf.score(tX, tY)))
    df = pd.read_csv(data_path + testinf)
    data = df.values
    tY = data[:, data.shape[1] - 1]
    # trans_T = train_data_T[:, 1:train_data_T.shape[1] - 1]
    tX = append_feature(df, istest=False)
    print("Inf score: {score}".format(score=clf.score(tX, tY)))
    print('****************')
    # *****************************************


# Compute MMD (maximum mean discrepancy) using numpy and scikit-learn (3 ways for MMD)
def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]
    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})
    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


# Here we recommend mmd_rbf for evaluating
# Current still exist wrong classifications which need to be fixed
def class_eval(data_path, ddos, bot, port, bruf, inf, test):
    """
    :param data_path: root path
    :param ddos: attack dataset type 1
    :param bot: attack dataset type 2
    :param port: attack dataset type 3
    :param bruf: attack dataset type 4
    :param inf: attack dataset type 5
    :param test: unlabelled instances (unknown types), NO LABELS HERE (np.array)
    :return: types from 0, 1, 2, 3, 4, 5 (0 for not sure type when calculating with MMD)
    Attack datasets 1-5 should contain single attack data only! To avoid mislead by normal type
    """
    ddos_X, ddos_Y = DataFrame_loader(data_path + ddos)
    bot_X, bot_Y = DataFrame_loader(data_path + bot)
    port_X, port_Y = DataFrame_loader(data_path + port)
    bruf_X, bruf_Y = DataFrame_loader(data_path + bruf)
    inf_X, inf_Y = DataFrame_loader(data_path + inf)
    score1 = mmd_rbf(test, ddos_X)
    score2 = mmd_rbf(test, bot_X)
    score3 = mmd_rbf(test, port_X)
    score4 = mmd_rbf(test, bruf_X)
    score5 = mmd_rbf(test, inf_X)
    score = np.array([score1, score2, score3, score4, score5])
    score = score / np.sum(score)
    x = score[abs(score - np.mean(score)) > 1.1 * np.std(score)]
    print(score, x)
    # Parametor: 1.25 -> 899; 1.5 -> 1081; 1.2 -> 827; 1.0 -> 780; 1.05 -> 699 (based on the datasets you use)
    # np.std(): Compute the standard deviation along the specified axis.
    # This 1.5 is tested by assuming 0.01 as the goal and as for the rest are at 0.75 & beyond. (100 vs 500)
    # Need to be changed based on the real test, for now, no other below 0.04 can have an output for the outlier.
    if x.size == 1:
        out = np.argmin(score) + 1
    else:
        out = 0  # Unknown types, probably most are 0 or no major types, use all models to classify
    return out


#  select small models for different purpose
def slim_model(path, T, clf0, clf1, clf2, clf3, clf4):
    """
    :param path: root path
    :param T: target dataset, single attack type for slim model
    :param clf0: updated model 0 (500)
    :param clf1: updated model 1 (500)
    :param clf2: updated model 2 (500)
    :param clf3: updated model 3 (500)
    :param clf4: updated model 4 (500)
    :return: specialized model for single attack (100)
    """
    clf = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)  # RF exp

    X, Y = DataFrame_loader(path + T)
    model = clf.fit(X, Y)
    # wts0 = clf0.estimator_weights_
    # wts1 = clf1.estimator_weights_
    # wts2 = clf2.estimator_weights_
    # wts3 = clf3.estimator_weights_
    # wts4 = clf4.estimator_weights_
    N = 50  # number of estimators in Random Forest (100)
    # est_wts = np.concatenate(5*N, axis=0)
    # est_wts = np.append(wts0, wts1)
    UpdatedBaseEst = []
    UpdatedBaseWet = np.zeros(N, )
    UpdatedErr = np.zeros(N, )
    score = np.zeros(5 * N)
    for i in range(0, N):
        score[i] = clf0.estimators_[i].score(X, Y)
    for i in range(N, 2 * N):
        score[i] = clf1.estimators_[i - N].score(X, Y)
    for i in range(2 * N, 3 * N):
        score[i] = clf2.estimators_[i - 2 * N].score(X, Y)
    for i in range(3 * N, 4 * N):
        score[i] = clf3.estimators_[i - 3 * N].score(X, Y)
    for i in range(4 * N, 5 * N):
        score[i] = clf4.estimators_[i - 4 * N].score(X, Y)
    rank = np.argsort(score)[::-1][0:10]  # 10 base_estimators only

    for i in range(rank.shape[0]):
        if rank[i] in range(0, N):
            model.estimators_[i].random_state = clf0.estimators_[rank[i]].random_state
        elif rank[i] in range(N, 2 * N):
            model.estimators_[i].random_state = clf1.estimators_[rank[i] - N].random_state
        elif rank[i] in range(2 * N, 3 * N):
            model.estimators_[i].random_state = clf2.estimators_[rank[i] - 2 * N].random_state
        elif rank[i] in range(3 * N, 4 * N):
            model.estimators_[i].random_state = clf3.estimators_[rank[i] - 3 * N].random_state
        else:
            model.estimators_[i].random_state = clf4.estimators_[rank[i] - 4 * N].random_state

    # model.estimators_ = UpdatedBaseEst
    # model.estimator_weights_ = UpdatedBaseWet
    # model.estimator_errors_ = UpdatedErr
    return model


# classify types and apply model (Main for multi-model classify)
def class_model(data_path, ddos, bot, port, bruf, inf, X, m0, m1, m2, m3, m4, m5):
    """
    :param data_path: root path
    :param ddos: pure ddos training dataset
    :param bot: pure bot training dataset
    :param port: pure port training dataset
    :param bruf: pure bruf training dataset
    :param inf: pure inf training dataset
    :param X: unlabelled dataset, waiting for classify major attack type and assign for models
    :param m0: updated model 500 for common type
    :param m1: ddos 100
    :param m2: bot 100
    :param m3: port 100
    :param m4: bruf 100
    :param m5: inf 100
    :return: predicted labels as np.array
    """
    pred = np.zeros(X.shape[0])
    for i in range(X.shape[0] // 100):
        if 100 * (i + 1) <= X.shape[0]:
            num = class_eval(data_path, ddos, bot, port, bruf, inf, X[100 * i: 100 * (i + 1), ])  # return model choice
            if num == 0:
                pred[100 * i: 100 * (i + 1), ] = m0.predict(X[100 * i: 100 * (i + 1), ])
            elif num == 1:
                pred[100 * i: 100 * (i + 1), ] = m1.predict(X[100 * i: 100 * (i + 1), ])
            elif num == 2:
                pred[100 * i: 100 * (i + 1), ] = m2.predict(X[100 * i: 100 * (i + 1), ])
            elif num == 3:
                pred[100 * i: 100 * (i + 1), ] = m3.predict(X[100 * i: 100 * (i + 1), ])
            elif num == 4:
                pred[100 * i: 100 * (i + 1), ] = m4.predict(X[100 * i: 100 * (i + 1), ])
            else:
                pred[100 * i: 100 * (i + 1), ] = m5.predict(X[100 * i: 100 * (i + 1), ])
        else:
            num = class_eval(data_path, ddos, bot, port, bruf, inf, X[100 * i:, ])
            if num == 0:
                pred[100 * i:, ] = m0.predict(X[100 * i:, ])
            elif num == 1:
                pred[100 * i:, ] = m1.predict(X[100 * i:, ])
            elif num == 2:
                pred[100 * i:, ] = m2.predict(X[100 * i:, ])
            elif num == 3:
                pred[100 * i:, ] = m3.predict(X[100 * i:, ])
            elif num == 4:
                pred[100 * i:, ] = m4.predict(X[100 * i:, ])
            else:
                pred[100 * i:, ] = m5.predict(X[100 * i:, ])

    return pred


def voting_agg(data_path, ddos, bot, port, bruf, inf, T, X, m0, m1, m2, m3, m4, m5, N):
    """
    This module aim to implement boosting algorithm in the federated model aggregation.
    :param data_path: root path
    :param ddos: pure ddos training dataset
    :param bot: pure bot training dataset
    :param port: pure port training dataset
    :param bruf: pure bruf training dataset
    :param inf: pure inf training dataset
    :param T: Training data
    :param X: unlabelled dataset, waiting for classify major attack type and assign for models
    :param m0: updated model 500 for common type
    :param m1: ddos 100
    :param m2: bot 100
    :param m3: port 100
    :param m4: bruf 100
    :param m5: inf 100
    :param N: number of per data block
    :return: predicted labels as np.array
    """
    Tx, Ty = DataFrame_loader(data_path + T)
    # random the dataset
    index = [i for i in range(len(Ty))]
    random.shuffle(index)
    Tx = Tx[index]
    Ty = Ty[index]

    pred = np.zeros(X.shape[0])
    for i in range(X.shape[0] // N):
        if N * (i + 1) <= X.shape[0]:
            wts = voting_weight(data_path, ddos, bot, port, bruf, inf, X[N * i: N * (i + 1), ])  # return weights
            print('DataBlock {i} >> DDoS:{a}; Bot:{b}; Port: {c}; Bruf: {d}; Inf: {e}; Avg: {f}'
                  .format(i=i+1, a=wts[0], b=wts[1], c=wts[2], d=wts[3], e=wts[4], f=wts[5]))

            t0 = time.time()
            eclf = VotingClassifier(estimators=[('DDoS', m1), ('Botnet', m2), ('PortScan', m3),
                                                ('BruteForce', m4), ('Infiltration', m5), ('Updated', m0)],
                                    voting='soft', weights=wts)
            eclf = eclf.fit(Tx, Ty)
            t1 = time.time()
            print('Time consumed for {n} packets: {t}'.format(n=N, t=t1-t0))
            pred[N * i: N * (i + 1), ] = eclf.predict(X[N * i: N * (i + 1), ])

        else:
            wts = voting_weight(data_path, ddos, bot, port, bruf, inf, X[N * i:, ])

            print('DataBlock {i} >> DDoS:{a}; Bot:{b}; Port: {c}; Bruf: {d}; Inf: {e}; Avg: {f}'
                  .format(i=i+1, a=wts[0], b=wts[1], c=wts[2], d=wts[3], e=wts[4], f=wts[5]))

            t0 = time.time()
            eclf = VotingClassifier(estimators=[('DDoS', m1), ('Botnet', m2), ('PortScan', m3),
                                                ('BruteForce', m4), ('Infiltration', m5), ('Updated', m0)],
                                    voting='soft', weights=wts)
            eclf = eclf.fit(Tx, Ty)
            t1 = time.time()
            print('Time consumed for {n} packets: {t}'.format(n=N, t=t1-t0))
            pred[N * i:, ] = eclf.predict(X[N * i:, ])

    return pred


def voting_weight(data_path, ddos, bot, port, bruf, inf, test):
    """
    :param data_path: root path
    :param ddos: attack dataset type 1
    :param bot: attack dataset type 2
    :param port: attack dataset type 3
    :param bruf: attack dataset type 4
    :param inf: attack dataset type 5
    :param test: unlabelled instances (unknown types), NO LABELS HERE (np.array)
    :return: weights of mmd_distance for voting (5 clients + 1)
    We use the following equation to return weights for voting, based on mmd_distance to return weights for each vote
    The averaged model use the averaged weight
    """
    ddos_X, ddos_Y = DataFrame_loader(data_path + ddos)
    bot_X, bot_Y = DataFrame_loader(data_path + bot)
    port_X, port_Y = DataFrame_loader(data_path + port)
    bruf_X, bruf_Y = DataFrame_loader(data_path + bruf)
    inf_X, inf_Y = DataFrame_loader(data_path + inf)
    score1 = mmd_rbf(test, ddos_X)
    score2 = mmd_rbf(test, bot_X)
    score3 = mmd_rbf(test, port_X)
    score4 = mmd_rbf(test, bruf_X)
    score5 = mmd_rbf(test, inf_X)
    score = np.array([score1, score2, score3, score4, score5])
    score = score / np.sum(score)
    # print('mmd score:', score)
    return np.append(-np.log10(2 * score),
                     -np.log10(2 * np.mean(score)))  # return 5 weights append one weight from mean
