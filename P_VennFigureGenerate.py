import  matplotlib.pyplot as plt
from matplotlib_venn import venn2
import re
import copy


DEGsPath = "d:\\P_LungDEGs.txt"
DCBTF_FilePath = "d:\\P_LungGenesList.txt"
DEGsSet = set()
DCBTF_Set = set()
with open(DEGsPath , mode="r") as fh :
    for line in fh:
        oneline = line.strip("\n")
        DEGsSet.add(oneline)
with open(DCBTF_FilePath,mode="r") as fh:
    for line in fh :
        oneline = line.strip("\n")
        splitList = re.split("\t",oneline)
        flag = int(splitList[1])
        if flag >= 3 :
            DCBTF_Set.add(splitList[0])
commonSet = set()
DEGsDistinction = set()
DCBTF_Distinction = copy.deepcopy(DCBTF_Set)
for geneName in DEGsSet :
    if DCBTF_Set.__contains__(geneName) is True :
        commonSet.add(geneName)
        DCBTF_Distinction.remove(geneName)
    else:
        DEGsDistinction.add(geneName)
DEGsLen = len(DEGsDistinction)
DCBTF_Len = len(DCBTF_Distinction)
print("DEGs length is " , DEGsLen)
print("DCBTF length is " , DCBTF_Len)
print("Commons genes length is " , len(commonSet))
v = venn2(subsets={"10":DEGsLen,"01":DCBTF_Len , "11" : len(commonSet)},
      set_labels=("Statistic Testing","Machine Learning"))
plt.title("The Venn Diagram Of Statistic And Machine Learning In Lung Cancer")
plt.show(v)


