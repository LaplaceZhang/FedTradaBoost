# -*- coding: UTF-8 -*-
import socket
from threading import Thread
import pickle

import numpy as np
import sklearn.metrics

from sklearn import tree
import Fed_main as tr
import Fed_AGG as agg
import pandas as pd
import random
import numpy
import os
from sklearn.ensemble import AdaBoostClassifier

"""
********************
-- THE CONTRIBUTE --
@author0: JIAZHEN ZHANG
@author1: xxx xxx
@author2: xxx xxx
********************
-- PROJECT README --
This is the main server of the federated learning system. Here we aim to use energy-constrained IoT devices to 
train a federated transfer learning module and get aggregation on this server. Considering limitations of IoT 
devices, public datasets should be small and the model need to be as light as possible, which implies the deep
learning may not be a good choice for such a system. Here we choose TrAdaBoost, using AdaBoost for base learner,
containing 100 base_estimators, run 30 iterations on each device, each round.
********************
-- BASIC WORKFLOW --
* RUN Fed_server.py, Fed_worker1.py ..
* SERVER - 1 for check number of connected devices
* SERVER - 4 for set up public samples to send (pickle file)
* CLIENT - 3 to check any received data (Always remember to print, this also capable to update received data files!)
* SERVER - 2 initialization a public model AdaBoost & send
* CLIENT - 3 to check any received data 
* CLIENT - 1 for receive model & start training
* SERVER - 3 send message to clients in case client not return
* CLIENT - 2 for send updated model back to server (pickle file)
* SERVER - 4 to re-allocate public samples from database (IF: use sample weights for data re-arrange; ELSE: skip this)
* SERVER - 5 for model aggregation, and return predicted results
* GO FOR NEXT ITERATION
* SERVER - 2 send updated model back to clients
* SERVER - 7 OR CLIENT - 5 to stop connection

ATTENTION: if any CLIENT got jammed, try use SERVER - 3 to send text message to target CLIENT
ATTENTION: always remember to use - 3 on CLIENT to write received data into your file
********************
-- PAPER SUPPORTS --
~No links here yet~
********************
-- IMPORT REQUIRE --
socket --> host server & clients, including message send and receive
threading --> multiple connections for server (only 2/3 clients used in this experiment)
pickle --> seal message to send and receive, including save to file and load
sklearn --> adaboost algorithm build, for base_estimators in our TrAdaBoost
Some basic --> numpy pandas Python-3.7&higher sklearn-0.24.2
********************
-- CONTACT ME VIA --
If you have any problems using this code, you may start an issue on project github page: 
~No links here yet~
********************
-- THE DISCLAIMER --
Code may contain bugs which may cause system crush (99.99% it won't), use at your own risk.
Also the overall performance is not guaranteed.
"""

ADDRESS = ('0.0.0.0', 7000)
g_socket_server = None  # 负责监听的socket
g_conn_pool = []  # connection pool
updates = []
num_public_instance = 10000  # DO NOT EXCEED THIS NUM TO CLIENT WHICH MAY CAUSE PICKLE ERROR!
data_path = "./server_save/data/"
model_path = "./server_save/model/"
all_in_one = "public.csv"
data_to_send = "ToSend.pickle"
weight0 = 'weight0.pickle'
weight1 = 'weight1.pickle'
weight2 = 'weight2.pickle'
weight3 = 'weight3.pickle'
weight4 = 'weight4.pickle'
saved_model = 'updatedmodel.pickle'
BaseDataset = 'Test.csv'  # 2500+1000+1000+1000+1000+1000
testbot = 'Bot.csv'
testddos = 'DDoS.csv'
testport = 'Port.csv'
testbruf = 'Bruf.csv'
testinf = 'Inf.csv'


def init():
    """
    Service Initialization
    """
    global g_socket_server
    tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    g_socket_server = socket.socket()  # build socket
    g_socket_server.bind(ADDRESS)
    g_socket_server.listen(5)  # awaiting num, not num of connections
    print("Server started, waiting...")


def accept_client():
    """
    Receive New Connections
    """
    while True:
        client, address = g_socket_server.accept()
        # Connection pool
        g_conn_pool.append(client)
        # independent thread
        thread = Thread(target=message_handle, args=(client,))
        thread.setDaemon(True)
        thread.start()


def message_handle(client):
    """
    Handle incoming messages
    """
    msg = "Hello from server!"
    msg = pickle.dumps(msg)
    client.sendall(msg)

    while True:
        received_data = b''
        while str(received_data)[-2] != '.':
            data = client.recv(409600)
            received_data += data
        received_data = pickle.loads(received_data)

        # identify weight feedback (update types: numpy.ndarray)
        if type(received_data) == numpy.ndarray:
            # if os.path.exists(data_path + saved_weights):  # True if .npy file exists
            #     old_data = numpy.load(data_path + saved_weights, allow_pickle=True)
            #     new_data = numpy.vstack((old_data, received_data))
            #     numpy.save(data_path + saved_weights, new_data)
            # else:
            #     numpy.save(data_path + saved_weights, received_data)
            print("Weight Received!")
            f = open('./server_save/data/weight{num}.pickle'.format(num=g_conn_pool.index(client)), 'wb')
            pickle.dump(received_data, f)
            f.close()
        elif type(received_data) == AdaBoostClassifier:
            print("Model Received!")
            f = open('./server_save/model/update{num}.pickle'.format(num=g_conn_pool.index(client)), 'wb')
            pickle.dump(received_data, f)
            f.close()
            print(received_data)
        else:
            print("Received from client{num}. Received information: {type} {received_data}".format(
                num=g_conn_pool.index(client),
                type=type(received_data), received_data=received_data))


if __name__ == '__main__':
    init()
    # set up a thread for main
    thread = Thread(target=accept_client)
    thread.setDaemon(True)
    thread.start()
    # main
    while True:
        cmd = input("""--------------------------
CMD1: Number of connected clients
CMD2: Send model to client
CMD3: Send text message to client
CMD4: Reallocate samples by weights
CMD5: Model aggregation 
CMD6: Model pretrain
CMD7: Stop all
""")
        if cmd == '1':
            print("--------------------------")
            print("Device connected：", len(g_conn_pool))

        elif cmd == '2':
            print("--------------------------")
            print("Create model & send to devices")
            # index = input("Input: number of client: ")
            # msg = tr.TrAdaBoost()
            '''
            Here we actually did not send the model to client, but it can use updated model 'stored in server'. 
            This is mainly because send updated model to each client is a little complex and in this simulation 
            experiment we build all server+client locally and client can directly use updated model without receiving 
            it again. We aggregate feedbacks in the server for model update and send command back to client so that 
            they can use it to train separately. 
            '''
            if os.path.exists(model_path + saved_model):
                f = open(model_path + saved_model, "rb")
                model = pickle.load(f)
                f.close()
                msg = pickle.dumps(model)
                # g_conn_pool[int(index)].sendall(msg)
                [g_conn_pool[i].sendall(msg) for i in range(len(g_conn_pool))]
                # print('Send model to client {num} & start training.'.format(num=index))
                print('Send model to all clients')
            # msg = pickle.dumps(msg)
            # g_conn_pool[int(index)].sendall(msg)
            # [g_conn_pool[i].sendall(msg) for i in range(len(g_conn_pool))]
            # print('Send model to client {num} & start training.'.format(num=index))
            # print('Send model to all clients')
            else:
                print("Model Initialization")
                clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2,
                                                                     min_samples_split=20,
                                                                     min_samples_leaf=5),
                                         algorithm="SAMME",
                                         n_estimators=500, learning_rate=0.5)
                X, Y = agg.DataFrame_loader(data_path + data_to_send)
                clf.fit(X, Y)

                X, Y = agg.DataFrame_loader(data_path + BaseDataset)
                print("Score for initialized model: {score}".format(score=clf.score(X, Y)))
                agg.PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, clf)
                msg = pickle.dumps(clf)
                # g_conn_pool[int(index)].sendall(msg)
                [g_conn_pool[i].sendall(msg) for i in range(len(g_conn_pool))]
                # print('Send model to client {num} & start training.'.format(num=index))
                print('Sent model to all clients')

        elif cmd == '3':
            print("--------------------------")
            index, msg = input("Input: number of client, text message").split(',')
            msg = pickle.dumps(msg)
            g_conn_pool[int(index)].sendall(msg)
            print('Send text to client {num}.'.format(num=index))

        elif cmd == '4':
            print("--------------------------")
            # index = input("Input: number of client: ")
            # if os.path.exists(data_path + data_to_send):  # True if .npy file exists
            #     print("Received weights of samples, re-arrange public data.")
            #     # Here I use pickle file to save weight, the avg.WeightSample need to be modified.
            #     samples = avg.WeightedSample(data_path, all_in_one, data_to_send, weight1, num_public_instance)
            #     samples.to_pickle(data_path + data_to_send)
            # else:
            print("Prepare public dataset.")
            # Initialization public dataset(csv file) and weights(npy file)
            samples = agg.Initialization(data_path, all_in_one, num_public_instance)
            samples.to_pickle(data_path + data_to_send)

            f = open(data_path + data_to_send, "rb")
            msg = pickle.load(f)
            f.close()
            print(msg)
            msg = pickle.dumps(msg)
            # g_conn_pool[int(index)].sendall(msg)
            [g_conn_pool[i].sendall(msg) for i in range(len(g_conn_pool))]
            print("Sent datasets to all clients.")
            # os.remove(data_path + saved_weights)

        elif cmd == '5':
            """
            This model aggregation aims to compare performance of base_estimators in each updated AdaBoost model. 
            """
            print("--------------------------")
            print("Model aggregation started.")
            f = open(model_path+"update0.pickle", "rb")
            m0 = pickle.load(f)
            f.close()
            f = open(model_path+"update1.pickle", "rb")
            m1 = pickle.load(f)
            f.close()
            f = open(model_path+"update2.pickle", "rb")
            m2 = pickle.load(f)
            f.close()
            f = open(model_path+"update3.pickle", "rb")
            m3 = pickle.load(f)
            f.close()
            f = open(model_path+"update4.pickle", "rb")
            m4 = pickle.load(f)
            f.close()
            """
            Required: 
            1. return weights of public samples --> re-allocate samples  
            2. returned adaboost model --> base classifier
            """

            X, Y = agg.DataFrame_loader(data_path + BaseDataset)

            if os.path.exists(data_path + data_to_send):
                print("Overall score from client0: {score}".format(score=m0.score(X, Y)))
                print("Overall score from client1: {score}".format(score=m1.score(X, Y)))
                print("Overall score from client2: {score}".format(score=m2.score(X, Y)))
                print("Overall score from client3: {score}".format(score=m3.score(X, Y)))
                print("Overall score from client4: {score}".format(score=m4.score(X, Y)))
                Xt, Yt = agg.DataFrame_loader(data_path + data_to_send)
                # updated_model = agg.FedRank(Xt, Yt, m0, m1, m2, m3, m4)  # Rank for aggregation
                updated_model = agg.FedAggDT(m0, m1, m2, m3, m4)  # Averaging for aggregation
                '''
                This should be the final output, no need for 500 estimators. 
                select only a few estimators may have a better results compared with all 500 enrolled. 
                example: public100 + private125
                '''
                print("Overall score from updated model: {score}".format(score=updated_model.score(X, Y)))

                print('m0: ')
                print(agg.PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, m0))
                print('m1: ')
                print(agg.PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, m1))
                print('m2: ')
                print(agg.PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, m2))
                print('m3: ')
                print(agg.PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, m3))
                print('m4: ')
                print(agg.PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, m4))
                print('updated: ')
                print(agg.PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, updated_model))

                f = open(model_path + saved_model, 'wb')
                pickle.dump(updated_model, f)
                f.close()

            else:
                print("Public samples are not exist, use CMD4 to create public data.")

        elif cmd == "6":
            print("Use specialized model for specific tasks")
            f = open(model_path+"update0.pickle", "rb")
            clf0 = pickle.load(f)
            f.close()
            f = open(model_path+"update1.pickle", "rb")
            clf1 = pickle.load(f)
            f.close()
            f = open(model_path+"update2.pickle", "rb")
            clf2 = pickle.load(f)
            f.close()
            f = open(model_path+"update3.pickle", "rb")
            clf3 = pickle.load(f)
            f.close()
            f = open(model_path+"update4.pickle", "rb")
            clf4 = pickle.load(f)
            f.close()
            f = open(model_path+saved_model, "rb")
            m0 = pickle.load(f)
            f.close()
            m1 = agg.slim_model(data_path, testddos, clf0, clf1, clf2, clf3, clf4)
            m2 = agg.slim_model(data_path, testbot, clf0, clf1, clf2, clf3, clf4)
            m3 = agg.slim_model(data_path, testport, clf0, clf1, clf2, clf3, clf4)
            m4 = agg.slim_model(data_path, testbruf, clf0, clf1, clf2, clf3, clf4)
            m5 = agg.slim_model(data_path, testinf, clf0, clf1, clf2, clf3, clf4)
            print("All slimmed models are done! Now return predicted labels for test")
            print('**********************')
            print('Print results for each attack models:')
            print('updated: ')
            print(agg.PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, m0))
            print('DDoS: ')
            print(agg.PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, m1))
            print('Botnet: ')
            print(agg.PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, m2))
            print('PortScan: ')
            print(agg.PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, m3))
            print('BruF.: ')
            print(agg.PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, m4))
            print('Inf.: ')
            print(agg.PrintAllAcc(data_path, testddos, testbot, testport, testbruf, testinf, m5))

            X, Y = agg.DataFrame_loader(data_path + BaseDataset)
            print('**********************')
            print("Overall score from client0: {score}".format(score=m0.score(X, Y)))
            print("Overall score from client1: {score}".format(score=m1.score(X, Y)))
            print("Overall score from client2: {score}".format(score=m2.score(X, Y)))
            print("Overall score from client3: {score}".format(score=m3.score(X, Y)))
            print("Overall score from client4: {score}".format(score=m4.score(X, Y)))
            print("Overall score from client5: {score}".format(score=m5.score(X, Y)))
            print('**********************')
            # random the dataset
            index = [i for i in range(len(Y))]
            random.shuffle(index)
            X = X[index]
            Y = Y[index]

            pred = agg.voting_agg(data_path, testddos, testbot, testport, testbruf, testinf, data_to_send,
                                   X, m0, m1, m2, m3, m4, m5)
            print(pred)
            print(sklearn.metrics.accuracy_score(Y, pred))

        elif cmd == '7':
            print("--------------------------")
            print("Server stopped.")
            msg = "Bye from server."
            msg = pickle.dumps(msg)
            [g_conn_pool[i].sendall(msg) for i in range(len(g_conn_pool))]
            exit()
