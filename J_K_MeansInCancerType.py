import re
import numpy as np
import sklearn.cluster.k_means_ as KMEANS

filePath = "d:\\StatisticResult_3.txt"
finalStorePath = "d:\\M_StomachFinalGenesList.txt"

kmeans = KMEANS.KMeans(n_clusters=6,n_init=6000,max_iter=60000)

geneList = []
clusterNumberList = []
countNumberList = []

print("Read data .")
with open(filePath , mode="r") as fh :
    for line in fh:
        oneLine = line.strip("\n")
        inforList = re.split("\t",oneLine)
        geneName = inforList[0]
        clusterNumber = inforList[1]
        countNumber = inforList[2]
        geneList.append(geneName)
        clusterNumberList.append(float(clusterNumber))
        countNumberList.append(float(countNumber))
print("Has been read .")
fitData = np.stack([clusterNumberList,countNumberList],axis=1)
kmeans.fit(fitData)
print("Completed")
centerList = []
print(kmeans.cluster_centers_)
for center in kmeans.cluster_centers_:
    print(center)
    centerList.append(float(center[0]))
sortedCenters = sorted(centerList)
print(centerList)
print(sortedCenters)
orFlag2FiFlag = {}
for i , sc in enumerate(sortedCenters):
    for j , oc in enumerate(centerList):
        if sc == oc :
            orFlag2FiFlag[j] = i
            continue
print(orFlag2FiFlag)
with open(finalStorePath,mode="w") as fh:
    for i , geneName in enumerate(geneList):
        fh.write(geneName  + "\t" + str(orFlag2FiFlag.get(kmeans.labels_[i])) + "\n")




