import sklearn.cluster.k_means_ as kMeans
import numpy as np
import os
import re

clusterNumber = 3
breastDic = "D:\ActivityMapping\\0"
liverDic = "D:\ActivityMapping\\1"
lungDic = "D:\ActivityMapping\\2"
stomachDic = "D:\ActivityMapping\\3"
storeResultDic = "d:\StoreResult"
filePath = "D:\ThesisDataDownLoad\LiverCancer\ExonExpression\TrainingSamples\TCGA-2V-A95S-01"
genesList = []
with open(filePath, mode="r") as f:
    for line in f:
        oneLine = line.strip("\n")
        geneName = re.split(pattern="\t", string=oneLine)[0]
        genesList.append(geneName)
print("The length of gene list is ", len(genesList))
kmeans = kMeans.KMeans(n_clusters=clusterNumber,max_iter=20000,n_init=346)
if os.path.exists(storeResultDic) is False:
    os.mkdir(storeResultDic)
dics = [breastDic,liverDic,lungDic,stomachDic]
for k , dic in enumerate(dics):
    for root , dictionaries , files in os.walk(dic) :
        for file in files:
            print(root)
            print(file)
            loadFile = re.match(pattern=".*(\\.npy)$",string=file)
            if loadFile is not None :
                fileName = loadFile[0]
                activationMap = np.load(os.path.join(root, fileName))
                activationArray = np.reshape(activationMap, newshape=[-1, 1])
                activationArray = np.squeeze(activationArray)
                activationArray = np.abs(activationArray)
                activationArray = np.reshape(activationArray,newshape=[-1,1])
                kmeans.fit(X=activationArray)
                storeName = fileName.strip(".npy")
                centers = []
                for i in range(clusterNumber):
                    centers.append(float(kmeans.cluster_centers_[i]))
                print(centers)
                sortedCenters = sorted(centers)
                print(sortedCenters)
                orFlag2FiFlag = {}
                for i,sCenterNumber in enumerate(sortedCenters):
                    for j , orCenterNumber in enumerate(centers):
                        if sCenterNumber == orCenterNumber :
                            orFlag2FiFlag[j] = i
                            continue
                print(orFlag2FiFlag)
                with open(os.path.join(storeResultDic,storeName + "_" + str(k) + ".txt"),mode="w") as h :
                    for i, geneName in enumerate(genesList):
                        if geneName != "NA":
                            h.write(geneName + "\t" + str(orFlag2FiFlag.get(kmeans.labels_[i])) + "\n")
### TCGA-A1-A0SI-01

