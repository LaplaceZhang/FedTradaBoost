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
Some basic --> numpy pandas Python 3.7.10
********************
-- CONTACT ME VIA --
If you have any problems using this code, you may start an issue on project github page: 
~No links here yet~
********************
-- THE DISCLAIMER --
Code may contain bugs which may cause system crush (99.99% it won't), use at your own risk.
Also the overall performance is not guaranteed.