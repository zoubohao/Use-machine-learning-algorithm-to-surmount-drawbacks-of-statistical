import os
import re
import shutil
import numpy as np

if __name__ == "__main__":
    width = 192
    height = 192
    file_root_path = "D:\ThesisDataDownLoad\LiverCancer"
    bExon = "D:\ThesisDataDownLoad\BreastCancer\ExonExpression\TCGA-A1-A0SB-01_ExonExpressionDataNew"
    bMethy = "D:\ThesisDataDownLoad\BreastCancer\Methylation\TCGA-A1-A0SB-01_Methylation450kDataNew"
    bRNA = "D:\ThesisDataDownLoad\BreastCancer\RNA-Seq\TCGA-A1-A0SB-01_GenesExpressionDataNew"
    liExon = "D:\ThesisDataDownLoad\LiverCancer\ExonExpression\TCGA-2V-A95S-01_ExonExpressionDataNew"
    liMethy = "D:\ThesisDataDownLoad\LiverCancer\Methylation\TCGA-2V-A95S-01_Methylation450kDataNew"
    liRNA = "D:\ThesisDataDownLoad\LiverCancer\RNA-Seq\TCGA-2V-A95S-01_GenesExpressionDataNew"
    luExon = "D:\ThesisDataDownLoad\LungCancer\ExonExpression\TCGA-05-4384-01_ExonExpressionDataNew"
    luMethy = "D:\ThesisDataDownLoad\LungCancer\Methylation\TCGA-05-4384-01_Methylation450kDataNew"
    luRNA = "D:\ThesisDataDownLoad\LungCancer\RNA-Seq\TCGA-05-4384-01_GenesExpressionDataNew"
    nExon = "D:\ThesisDataDownLoad\\NormalSample\ExonExpression\TCGA-22-5471-11_ExonExpressionDataNew"
    nMethy = "D:\ThesisDataDownLoad\\NormalSample\Methylation\TCGA-22-5471-11_Methylation450kDataNew"
    nRNA = "D:\ThesisDataDownLoad\\NormalSample\RNA-Seq\\TCGA-22-5471-11_GenesExpressionDataNew"
    sExon = "D:\ThesisDataDownLoad\StomachCancer\ExonExpression\TCGA-B7-5816-01_ExonExpressionDataNew"
    sMethy = "D:\ThesisDataDownLoad\StomachCancer\Methylation\TCGA-B7-5816-01_Methylation450kDataNew"
    sRNA = "D:\ThesisDataDownLoad\StomachCancer\RNA-Seq\TCGA-B7-5816-01_GenesExpressionDataNew"
    pathList = [bExon,bMethy,bRNA,
                liExon,liMethy,liRNA,
                luExon,luMethy,luRNA,
                nExon,nMethy,nRNA,
                sExon,sMethy,sRNA]
    allGenesSet = set()
    for filePath in pathList:
        with open(filePath,mode="r") as f :
            for i , line in enumerate(f):
                lineS = line.strip("\n")
                lineSplit = re.split("\t",lineS)
                allGenesSet.add(lineSplit[0])
    print(len(allGenesSet))
    # genesList = list(allGenesSet)
    # paddingNum = width * height - len(genesList)
    # if paddingNum % 2 == 0 :
    #     paddingGeneList = ["NA" for i in range(paddingNum // 2)] + genesList\
    #                       + ["NA" for i in range(paddingNum // 2)]
    # else:
    #     paddingGeneList = ["NA" for i in range(paddingNum // 2)] + genesList \
    #                       + ["NA" for i in range(paddingNum // 2 + 1)]
    # print(len(paddingGeneList))
    #
    #
    # def CreateSample(root_dic,outputPathDic,geneList):
    #     for root , dics , files in os.walk(root_dic):
    #         print(dics)
    #         for file in files:
    #             if re.match(pattern="^TCGA-.*New",string=file):
    #                 print(file)
    #                 gene2value = {}
    #                 with open(os.path.join(root,file),mode="r") as fh:
    #                     for oneLine in fh:
    #                         thisLine = re.split("\t",oneLine.strip("\n"))
    #                         geneName = thisLine[0]
    #                         geneValue = thisLine[1]
    #                         gene2value[geneName] = geneValue
    #                 if os.path.exists(os.path.join(root,outputPathDic)) is False:
    #                     os.mkdir(os.path.join(root,outputPathDic))
    #                 # with open(os.path.join(root,outputPathDic,file),mode="w") as fh:
    #                 #     for oneGene in geneList:
    #                 #         if oneGene == "NA":
    #                 #             fh.write("NA" + "\t" + str(0.0) + "\n")
    #                 #         else:
    #                 #             if gene2value.__contains__(oneGene) is False:
    #                 #                 fh.write(oneGene + "\t" + str(0.0) + "\n")
    #                 #             else:
    #                 #                 fh.write(oneGene + "\t" + str(gene2value.get(oneGene)) + "\n")
    #         for dic in dics:
    #             CreateSample(os.path.join(root,dic),outputPathDic,geneList)
    #
    #
    # CreateSample(file_root_path, "FinalSamplesDic",paddingGeneList)









