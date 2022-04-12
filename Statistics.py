import os
import sys

f = open("/data/tongji/1/k0.txt")
line = f.readline()  
j=0
count=[0 for i in range(1000) ]
while line:  
    print (line)  
    line = f.readline()
    o=float(line)/0.1
    # if j>696:
    count[int(o)]+=1
    j+=1
    # if float(line)<0.1:
    #     count[0]+=1
    # elif float(line)<0.1:
    #     count[0]+=1
        

f.close()
