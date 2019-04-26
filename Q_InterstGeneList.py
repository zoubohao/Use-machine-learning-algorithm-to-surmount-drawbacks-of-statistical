import re

### Filter the clustered class equal or bigger than 4
fileName = "d:\\M_StomachFinalGenesList.txt"
storeFile = "d:\\N_StomachStrongRelatedGenesList.txt"
### Filter some of genes which are not in the annotation file
goFile = "d:\\goa_human.gaf"
inputFile = "d:\\N_StomachStrongRelatedGenesList.txt"
outputFile = "d:\\N_StomachModifyStrongRelated.txt"
with open(fileName,mode="r") as fh:
    with open(storeFile,mode="w") as wh:
        for line in fh:
            oneLine = line.strip("\n")
            geneName = re.split("\t", oneLine)[0]
            classValue = int(re.split("\t", oneLine)[1])
            if classValue >=4 :
                wh.write(geneName + "\n")
allGenesSet = set()
with open(goFile , mode="r") as fh :
    for line in fh:
        oneLine = line.strip("\n")
        inforList = re.split("\t",oneLine)
        geneName = inforList[2]
        allGenesSet.add(geneName)
with open(inputFile,mode="r") as fh :
    with open(outputFile,mode="w") as oh :
        for line in fh :
            oneLine = line.strip("\n")
            if oneLine in allGenesSet :
                oh.write(oneLine + "\n")

