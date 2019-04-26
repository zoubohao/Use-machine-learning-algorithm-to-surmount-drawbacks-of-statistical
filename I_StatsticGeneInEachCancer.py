import os
import re


filesDic = "D:\StoreResult"

breastMap = {}
liverMap = {}
lungMap = {}
stomachMap = {}
breastCount = {}
liverCount = {}
lungCount = {}
stomachCount = {}
mapList = [breastMap,liverMap,lungMap,stomachMap]
countMapList = [breastCount,liverCount,lungCount,stomachCount]

k = 0
for root , dics , files in os.walk(filesDic) :
    for file in files:
        print(k)
        print(file)
        cancerFlag = int(re.split("_",file.strip(".txt"))[1])
        with open(os.path.join(root,file),mode="r") as fh:
            for line in fh:
                oneLine = line.strip("\n")
                geneName = re.split("\t",oneLine)[0]
                clusterClass = int(re.split("\t",oneLine)[1])
                thisMap = mapList[cancerFlag]
                thisCountMap = countMapList[cancerFlag]
                if thisMap.__contains__(geneName) is False:
                    thisMap[geneName] = clusterClass
                    if clusterClass != 0 :
                        thisCountMap[geneName] = 1
                    else:
                        thisCountMap[geneName] = 0
                else:
                    value = thisMap.get(geneName)
                    value = value + clusterClass
                    thisMap[geneName] = value
                    if clusterClass != 0:
                        count = thisCountMap.get(geneName)
                        count += 1
                        thisCountMap[geneName] = count
        k += 1

for i,cancerMap in enumerate(mapList):
    with open("d:\\StatisticResult_" + str(i) + ".txt",mode="w") as fh:
        for key , value in cancerMap.items():
            fh.write(key + "\t" + str(value) + "\t" + str(countMapList[i].get(key)) + "\n")

