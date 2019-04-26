import scipy.stats as stats
import numpy as np
import re
import math


matrixFilePath = "D:\ThesisDataDownLoad\StomachCancer\Stomach Cancer Gene Expression RNA-Seq Data"
DEGsStorePath = "D:\\StomachDEGs.txt"
threshold = 0.01
tumorSamplesLocation = []
normalSamplesLocation = []
PvaluesList = []
genesNameList = []

with open(matrixFilePath,mode="r") as fh :
    for i, line in enumerate(fh):
        print(i)
        oneLine = line.strip("\n")
        if i == 0:
            samplesList = re.split("\t", oneLine)
            for k, sampleName in enumerate(samplesList):
                if k != 0:
                    flag = int(re.split("-", sampleName)[-1])
                    if flag >= 10:
                        normalSamplesLocation.append(k)
                    else:
                        tumorSamplesLocation.append(k)
            print(len(tumorSamplesLocation))
            print(len(normalSamplesLocation))
        else:
            dataList = re.split("\t", oneLine)
            print(dataList[0])
            tumorData = []
            normalData = []
            for loca in normalSamplesLocation:
                normalData.append(float(dataList[loca]))
            for loca in tumorSamplesLocation:
                tumorData.append(float(dataList[loca]))
            tumorData = np.array(tumorData, dtype=np.float32)
            normalData = np.array(normalData, dtype=np.float32)
            ### normal distribution test
            tNormalPValue = stats.shapiro(tumorData)[1]
            nNormalPValue = stats.shapiro(normalData)[1]
            print("The p value of tumor samples for testing normalization distribution is ", tNormalPValue)
            print("The p value of normal samples for testing normalization distribution is ", nNormalPValue)
            if tNormalPValue <= threshold or nNormalPValue <= threshold:
                finalPValue = stats.ranksums(tumorData, normalData)[1]
                print("The final p-value of rank sums test is  ", finalPValue)
                PvaluesList.append(finalPValue)
                genesNameList.append(dataList[0])
            else:
                equalVarPValue = stats.levene(tumorData, normalData)[1]
                print("The p-value of F-test is ", equalVarPValue)
                if equalVarPValue <= threshold:
                    finalPValue = stats.ttest_ind(tumorData, normalData, equal_var=False)[1]
                    print("The final p-value of T-test is  ", finalPValue)
                    PvaluesList.append(finalPValue)
                    genesNameList.append(dataList[0])
                else:
                    finalPValue = stats.ttest_ind(tumorData, normalData, equal_var=True)[1]
                    print("The final p-value of T-test is  ", finalPValue)
                    PvaluesList.append(finalPValue)
                    genesNameList.append(dataList[0])

def FDR_Correction(pvalueList,geneList,th,outFilePath) :
    length = len(pvalueList)
    newPvalueList = []
    newGenesList = []
    for n in range(length) :
        if math.isnan(pvalueList[n]) is False:
            newPvalueList.append(pvalueList[n])
            newGenesList.append(geneList[n])
    print("Sorting")
    newLength = len(newPvalueList)
    print(newLength)
    for pos in range(newLength) :
        for g in range(newLength - pos - 1):
            if newPvalueList[g+1] < newPvalueList[g] :
                temp = newPvalueList[g+1]
                newPvalueList[g+1] = newPvalueList[g]
                newPvalueList[g] = temp
                gTemp = newGenesList[g+1]
                newGenesList[g+1] = newGenesList[g]
                newGenesList[g] = gTemp
    print("Writting")
    with open(outFilePath,mode="w") as wh :
        for l in range(newLength):
            q_value = (newPvalueList[l] * (newLength / (l + 1.0)) + 0.0)
            print(newGenesList[l])
            print(q_value)
            if q_value < th :
                wh.write(newGenesList[l] + "\n")
FDR_Correction(pvalueList=PvaluesList,geneList=genesNameList,th = threshold , outFilePath=DEGsStorePath)













