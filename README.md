## Federated learning with transfer approach in IDS for IoT/IIoT scenarios

[@LaplaceZhang](https://github.com/LaplaceZhang) 

[@marcusCarpenter97](https://github.com/marcusCarpenter97)

***

### PROJECT NOTICE

This is the main server of the federated learning system. Here we aim to use energy-constrained IoT devices to 
train a federated transfer learning module and get aggregation on this server. Considering limitations of IoT 
devices, public datasets should be small and the model need to be as light as possible, which implies the deep
learning may not be a good choice for such a system. Here we choose TrAdaBoost, using AdaBoost for base learner,
containing 100 base_estimators, run 30 iterations on each device, each round.

***

### BASIC WORKFLOW

* START SERVER & CLIENT
* USE following commands:
* __SERVER__ - 1 for check number of connected devices
* __SERVER__ - 4 for set up public samples to send (:cucumber: *pickle file*)
* __CLIENT__ - 3 to print any received data
* __SERVER__ - 2 initialization a public model AdaBoost & distribute to __CLIENT__
* __CLIENT__ - 3 to print any received data 
* __CLIENT__ - 1 to start training TrAdaBoost _OR_  __CLIENT__ - 4 to start training AdaBoost (_NO TRANSFER_)
* __SERVER__ - 3 to send message to __CLIENT__ in case client not return
* __CLIENT__ - 2 for send updated model back to server (:cucumber: *pickle file*)
* __SERVER__ - 4 to re-allocate public samples from database (__IF__: use sample weights for data re-arrange; __ELSE__: skip this)
* __SERVER__ - 5 for model aggregation (__FedAvg__ *or* __Rank__), and return results using multiple binary datasets to test, as well as the overall score
* __SERVER__ - 6 for aggregation with __Weighted Voting__
* __SERVER__ - 2 send updated model back to clients
* __SERVER__ - 7 stop connection

:exclamation: If terminal got jammed, try use __SERVER__ - 3 to send text message to target __CLIENT__

:exclamation: Always remember to use __CLIENT__ - 3 to write received data into your file

***
### DATA CONFIG 

| Configurations | Attack Types with Percentage                                               | Total Packets |
|----------------|----------------------------------------------------------------------------|---------------|
| Public (SD)    | DDoS-17(20%), Botnet-17(20%), PortScan(20%),  BruteForce(10%), Infiltration(10%) | 10,000        |
| Client1 (TD1)  | DDoS-18(10%), Botnet-18(10%)                                                     | 4,400         |
| Client2 (TD2)  | DDoS-18(10%), Botnet-18(5%), BruteForce(10%),Infiltration(15%)                   | 5,000         |
| Client3 (TD3)  | DDoS-17(10%), Botnet-17(5%),PortScan(10%)                                        | 4,500         |
| Client4 (TD4)  | PortScan(5%), Infiltration(10%), BruteForce(10%)                           | 5,760         |
| Client5 (TD5)  | BruteForce(10%), Infiltration(10%)                                         | 4,000         |


***

### PAPER SUPPORT

Not released yet :no_entry_sign:

***

### To contact

If you have any problems using this code, you may raise an issue on [GITHUB](https://github.com/LaplaceZhang/FedTradaBoost).
