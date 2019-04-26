import re
import copy
import math



allGenesAnnotationPath = "d:\\goa_human.gaf"
DEGsGenesPath = "d:\\StomachDEGs.txt"
DCBTF_GensPath = "d:\\M_StomachFinalGenesList.txt"
DCBTF_Classes_Num = 6
allGenes2IDs = {}
DCBTF_Genes_Sets_List = []
with open(allGenesAnnotationPath , mode="r") as fh :
    for line in fh:
        oneLine = line.strip("\n")
        inforList = re.split("\t",oneLine)
        #print(inforList)
        geneName = inforList[2]
        idGo = inforList[4]
        processFlag = inforList[8]
        if geneName not in allGenes2IDs:
            idList = set()
            if processFlag == "P":
                idList.add(idGo)
            allGenes2IDs[geneName] = idList
        else:
            idList = allGenes2IDs.get(geneName)
            if processFlag == "P":
                idList.add(idGo)
            allGenes2IDs[geneName] = idList
allGenesSet = set(allGenes2IDs.keys())
DCBTF_Genes_Sets_List.append(copy.deepcopy(allGenesSet))
DEGsGenesSet = set()
Non_DEGsGenesSet = copy.deepcopy(allGenesSet)
for i in range(DCBTF_Classes_Num - 1):
    DCBTF_Genes_Sets_List.append(set())
with open(DEGsGenesPath,mode="r") as fh:
    for line in fh:
        oneLine = line.strip("\n")
        if oneLine in allGenesSet:
            DEGsGenesSet.add(oneLine)
            Non_DEGsGenesSet.remove(oneLine)
with open(DCBTF_GensPath,mode="r") as fh:
    for line in fh:
        oneLine = line.strip("\n")
        inforList = re.split("\t",oneLine)
        geneName = inforList[0]
        classOfGene = int(inforList[1])
        if geneName in allGenesSet:
            if classOfGene != 0 :
                DCBTF_Genes_Sets_List[classOfGene].add(geneName)
                DCBTF_Genes_Sets_List[0].remove(geneName)
print(len(DEGsGenesSet))
print(len(Non_DEGsGenesSet))
print("***********")
for oneSet in DCBTF_Genes_Sets_List:
    print(len(oneSet))

def entropyCalculate(probability):
    result = - probability * math.log2(probability)
    return result

GO_ID2Count = {}
allGenesAnnotationTotalCount = 0
for key , value in allGenes2IDs.items():
    for goID in value:
        if goID not in GO_ID2Count :
            GO_ID2Count[goID] = 1
        else:
            count = GO_ID2Count.get(goID)
            count += 1
            GO_ID2Count[goID] = count
        allGenesAnnotationTotalCount += 1
print("**********")
allIDsEntropy = 0.
for key , value in GO_ID2Count.items():
    allIDsEntropy = allIDsEntropy + entropyCalculate(float(value) / float(allGenesAnnotationTotalCount))
print("All genes set entropy is ",allIDsEntropy)

def ThisSetEntropyCalculate(thisSet):
    this_Go_ID2Count = {}
    thisTotalCount = 0
    for thisName in thisSet:
        thisGOIDList = allGenes2IDs.get(thisName)
        for thisGoID in thisGOIDList:
            if thisGoID not in this_Go_ID2Count:
                this_Go_ID2Count[thisGoID] = 1
            else:
                thisCount = this_Go_ID2Count.get(thisGoID)
                thisCount += 1
                this_Go_ID2Count[thisGoID] = thisCount
            thisTotalCount += 1
    thisEntropy = 0.
    for thiskey , thisvalue in this_Go_ID2Count.items():
        thisEntropy = thisEntropy + entropyCalculate(float(thisvalue) / float(thisTotalCount))
    return thisEntropy , thisTotalCount
### Calculate DEGs entropy .
DEGsEntropy , DEGsTotalCount = ThisSetEntropyCalculate(DEGsGenesSet)
### Calculate Non-DEGs entropy
Non_DEGsGenesEntropy , Non_DEGsTotalCount = ThisSetEntropyCalculate(Non_DEGsGenesSet)
statisticInforGain = allIDsEntropy - (DEGsTotalCount / allGenesAnnotationTotalCount + 0.) * DEGsEntropy - \
                     (Non_DEGsTotalCount / allGenesAnnotationTotalCount + 0.) * Non_DEGsGenesEntropy
print("Statistic method entropy is " ,(DEGsTotalCount / allGenesAnnotationTotalCount + 0.) * DEGsEntropy + \
                     (Non_DEGsTotalCount / allGenesAnnotationTotalCount + 0.) * Non_DEGsGenesEntropy)
print("Statistic method information gain is ",statisticInforGain)
splitStaInfor = entropyCalculate(float(DEGsTotalCount) / float(allGenesAnnotationTotalCount)) + \
             entropyCalculate(float(Non_DEGsTotalCount) / float(allGenesAnnotationTotalCount))
print(DEGsTotalCount + Non_DEGsTotalCount )
print("All genes total count is " , allGenesAnnotationTotalCount)
print("split sta infor " , splitStaInfor)
print("Statistic Information gain ratio is ",statisticInforGain / splitStaInfor)
### Deep
deepEntropy = 0.
thisDeepCountList = []
splitDeepInfor  = 0.
for deepSet in DCBTF_Genes_Sets_List:
    thisDeepEntropy , thisDeepCount = ThisSetEntropyCalculate(deepSet)
    splitDeepInfor = splitDeepInfor + entropyCalculate(float(thisDeepCount) / float(allGenesAnnotationTotalCount))
    thisDeepCountList.append(thisDeepCount)
    factor = thisDeepCount / allGenesAnnotationTotalCount + 0.
    deepEntropy = deepEntropy + factor * thisDeepEntropy
print("Deep learning entropy is ",deepEntropy)
print("Deep learning information gain is " , allIDsEntropy - deepEntropy)
print("split deep infor " , splitDeepInfor)
print("Deep Information gain ratio is ",(allIDsEntropy - deepEntropy) / splitDeepInfor)
print(sum(thisDeepCountList))
print("Improved ration : ",(((allIDsEntropy - deepEntropy) / splitDeepInfor) - (statisticInforGain / splitStaInfor)) / (statisticInforGain / splitStaInfor) + 0.0)