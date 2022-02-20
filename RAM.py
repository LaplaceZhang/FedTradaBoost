import os
import time
import re
import pandas as pd
import csv

with open("ram5.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Total memory", "Used memory", "Memory usage"])

while True:
    # get memory usage
    cpuinfo = os.popen("free -m | grep Mem")
    strMem = cpuinfo.readline()

    # get all info
    numMem = re.findall("\d+", strMem)
    total = numMem[0]
    used = numMem[1]
    usage = int(numMem[1])*100/int(numMem[0])

    print("[CLIENT5] Total memory: %sMB, Used memory: %sMB, Memory usage: %d%%"\
            %(total, used, usage))

    with open("ram5.csv", "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([[total, used, usage]])

    time.sleep(30)
