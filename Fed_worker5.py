# -*- coding: UTF-8 -*-
import socket
import pickle
import os
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
import Fed_main as tr
# from sklearn.impute import SimpleImputer

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import LocalTrain5 as LTtrain

root_path = "./local_save/client5/"
PublicData = "Received.csv"
Feedback = "model.pickle"
Received = "received.pickle"
Weights = "weights.npy"

soc = socket.socket()
print("Socket is created.")

ADDRESS = ('144.173.65.106', 7000)
# ADDRESS = ('127.0.0.1', 7000)
soc.connect(ADDRESS)
print("Connected to the server.")

msg = 'Hello from client2'
msg = pickle.dumps(msg)
soc.sendall(msg)
print("--------------------------")


def message_handle():
    """
    消息处理
    """
    # received_data = b''
    while True:
        received_data = b''
        while str(received_data)[-2] != '.':
            data = soc.recv(409600)
            received_data += data
        received_data = pickle.loads(received_data)
        print("Received information: {received_data}".format(received_data=received_data))

        if type(received_data) == pd.DataFrame:
            if os.path.exists(root_path + PublicData):
                os.remove(root_path + PublicData)
                received_data.to_csv(root_path + PublicData, index=False)
            else:
                received_data.to_csv(root_path + PublicData, index=False)
        elif type(received_data) == AdaBoostClassifier:
            print('model received')
            f = open(root_path + Received, 'wb')
            pickle.dump(received_data, f)
            f.close()
        return received_data


if __name__ == '__main__':
    while True:
        # received_data = b''
        # while str(received_data)[-2] != '.':
        #     data = soc.recv(1024)
        #     received_data += data
        # received_data = pickle.loads(received_data)
        # print("Received information: {received_data}".format(received_data=received_data))
        # Main CMD
        # Use client-cmd5 to exit
        received_data = message_handle()
        cmd = input("""--------------------------
CMD1: Process received data 
CMD2: Print feedback
CMD3: print any message sent from server
CMD4: Directly use classifier as AdaBoost
CMD5: stop server
""")

        if cmd == '1':
            print("--------------------------")
            print("Process collected information")
            # try:
            Source_data, Test_data, Source_label, Test_label, Target_data, Target_label = LTtrain.process()
            f = open(root_path + Received, 'rb')
            clf = pickle.load(f)
            f.close()
            print(clf)
            #  show the bench mark of clf
            # clf.fit(Source_data, Source_label)
            print("Benchmark score: {score}".format(score=clf.score(Test_data, Test_label)))
            # try:
            #     wts = np.load(root_path + Weights).T
            # except:
            row_S = Source_data.shape[0]  # X_train for source, trans_T for target, X_test for test
            wts = np.ones(row_S) / row_S
            pred, updated_model, wts = tr.TrAdaBoost().tradaboost(wts, clf, Source_data,
                                                                  Target_data, Source_label, Target_label, Test_data,
                                                                  Test_label, 15)

            # write updated_model into a pickle file
            f = open(root_path + Feedback, 'wb')
            pickle.dump(updated_model, f)
            f.close()
            fpr, tpr, thresholds = metrics.roc_curve(y_true=Test_label, y_score=pred, pos_label=0)
            print(updated_model)
            print('auc:', metrics.auc(fpr, tpr))
            print("acc:", accuracy_score(pred, Test_label))
            msg = wts
            msg = pickle.dumps(msg)
            soc.sendall(msg)

            try:
                os.remove(root_path + Weights)
                np.save(root_path + Weights, wts)
            except:
                np.save(root_path + Weights, wts)

        elif cmd == '2':
            print("--------------------------")
            print("Send model back to server.")
            f = open(root_path + Feedback, 'rb')
            model = pickle.load(f)
            f.close()
            msg = model
            msg = pickle.dumps(msg)
            soc.sendall(msg)

        elif cmd == '3':
            print("--------------------------")
            print("Printing Received Data")
            print(received_data)

        elif cmd == '4':
            print("--------------------------")
            print("Process collected information")
            # try:
            Source_data, Test_data, Source_label, Test_label, Target_data, Target_label = LTtrain.process()
            f = open(root_path + Received, 'rb')
            clf = pickle.load(f)
            f.close()
            print(clf)
            #  show the bench mark of clf
            # clf.fit(Target_data, Target_label)
            print("Benchmark score: {score}".format(score=clf.score(Test_data, Test_label)))
            # try:
            #     wts = np.load(root_path + Weights).T
            #
            # except:
            row_S = Source_data.shape[0]  # X_train for source, trans_T for target, X_test for test
            wts = np.ones(row_S) / row_S

            # Implement AdaBoost, directly use clf for detection, add iterations (?)
            updated_model = clf.fit(Target_data, Target_label)
            pred = clf.predict(Test_data)
            print('Score for Source', clf.score(Source_data, Source_label))
            print('Score for Target', clf.score(Target_data, Target_label))
            print('Score for Test', clf.score(Test_data, Test_label))

            # write updated_model into a pickle file
            f = open(root_path + Feedback, 'wb')
            pickle.dump(updated_model, f)
            f.close()
            fpr, tpr, thresholds = metrics.roc_curve(y_true=Test_label, y_score=pred, pos_label=0)
            print(updated_model)
            print('auc:', metrics.auc(fpr, tpr))
            print("acc:", accuracy_score(pred, Test_label))
            msg = wts
            msg = pickle.dumps(msg)
            soc.sendall(msg)

        elif cmd == '5':
            print("--------------------------")
            print("Client exit")
            exit()
