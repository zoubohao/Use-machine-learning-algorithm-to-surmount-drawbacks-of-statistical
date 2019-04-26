import re

goFile = "d:\\goa_human.gaf"
cancerFinalListPath = "d:\\M_LungFinalGenesList.txt"
outputGenesListPath = "d:\\P_LungGenesList.txt"
DEGsPath = "d:\\LungDEGs.txt"
outputDEGsPath = "d:\\P_LungDEGs.txt"

allGenesSet = set()
with open(goFile , mode="r") as fh :
    for line in fh:
        oneLine = line.strip("\n")
        inforList = re.split("\t",oneLine)
        geneName = inforList[2]
        allGenesSet.add(geneName)
with open(cancerFinalListPath , mode="r") as fh :
    with open(outputGenesListPath,mode="w") as wh:
        for line in fh:
            oneLine = line.strip("\n")
            inforList = re.split("\t", oneLine)
            geneName = inforList[0]
            classC = inforList[1]
            if geneName in allGenesSet:
                wh.write(geneName + "\t" + classC + "\n")
with open(DEGsPath,mode="r") as fh:
    with open(outputDEGsPath,mode="w") as wh:
        for line in fh:
            geneName = line.strip("\n")
            if geneName in allGenesSet:
                wh.write(geneName + "\n")




