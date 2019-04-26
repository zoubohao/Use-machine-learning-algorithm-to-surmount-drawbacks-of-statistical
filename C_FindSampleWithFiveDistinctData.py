import os
import re
import shutil

if __name__ == "__main__":

    ####### Cancer samples code
    # root_Dic = "D:\ThesisDataDownLoad\StomachCancer"
    # normal_root_Dic = "D:\ThesisDataDownLoad\\NormalSample"
    # data_type_map = {"CopyNumberData" : "CopyNumberData",
    #                  "ExonExpressionData":"ExonExpression",
    #                  "GenesExpressionData":"RNA-Seq",
    #                  "Methylation450kData":"Methylation",
    #                  "SomaticMutationData":"SomaticMutation"}
    # FinalMappingResult = {}
    # def Find_Files(root_Dict,mappingData,finalMap,normal_dic):
    #     for root, dics, files in os.walk(root_Dict):
    #         for dic in dics:
    #             Find_Files(os.path.join(root,dic),mappingData,finalMap,normal_dic)
    #         for file in files:
    #             if re.match(pattern="^(TCGA)-.{2}-.{4}-.{2}_.*",string=file):
    #                 TCGAid = re.split(pattern="_",string=file)[0]
    #                 dataType = re.split(pattern="_",string=file)[1]
    #                 judgeNum = re.split(pattern="-",string=TCGAid)[-1]
    #                 judgeNum = int(judgeNum)
    #                 if judgeNum > 10 :
    #                     shutil.move(os.path.join(root,file),
    #                                 os.path.join(normal_dic,mappingData[dataType],file))
    #                 if finalMap.__contains__(TCGAid) is False:
    #                     k = 1
    #                     finalMap[TCGAid] = k
    #                     print(os.path.join(root))
    #                     print(TCGAid + " : " + str(k))
    #                 else:
    #                     k = finalMap[TCGAid]
    #                     k = k + 1
    #                     finalMap[TCGAid] = k
    #                     print(os.path.join(root))
    #                     print(TCGAid + " : " + str(k))
    # Find_Files(root_Dic,data_type_map,FinalMappingResult,normal_root_Dic)
    # commonFileList = []
    # for key , value in FinalMappingResult.items():
    #     if value >= 10 :
    #         commonFileList.append(key)
    # for key , value in data_type_map.items():
    #     for roots , dicss , filess in os.walk(os.path.join(root_Dic,value)):
    #         for tcgaFile in filess:
    #             tcgaID = re.split(pattern="_",string=tcgaFile)[0]
    #             if commonFileList.__contains__(tcgaID) is False:
    #                 os.remove(os.path.join(roots,tcgaFile))

    ####### Normal samples code
    root_Dic = "D:\ThesisDataDownLoad\\NormalSample"
    data_type_map = {"CopyNumberData" : "CopyNumberData",
                     "ExonExpressionData":"ExonExpression",
                     "GenesExpressionData":"RNA-Seq",
                     "Methylation450kData":"Methylation",
                     "SomaticMutationData":"SomaticMutation"}
    FinalMappingResult = {}
    def Find_Files(root_Dict,mappingData,finalMap):
        for root, dics, files in os.walk(root_Dict):
            for dic in dics:
                Find_Files(os.path.join(root,dic),mappingData,finalMap)
            for file in files:
                if re.match(pattern="^(TCGA)-.{2}-.{4}-.{2}_.*",string=file):
                    TCGAid = re.split(pattern="_",string=file)[0]
                    if finalMap.__contains__(TCGAid) is False:
                        k = 1
                        finalMap[TCGAid] = k
                        print(os.path.join(root))
                        print(TCGAid + " : " + str(k))
                    else:
                        k = finalMap[TCGAid]
                        k = k + 1
                        finalMap[TCGAid] = k
                        print(os.path.join(root))
                        print(TCGAid + " : " + str(k))
    Find_Files(root_Dic,data_type_map,FinalMappingResult)
    commonFileList = []
    for key , value in FinalMappingResult.items():
        if value >= 6 :
            commonFileList.append(key)
    print(len(commonFileList))
    for key , value in data_type_map.items():
        for roots , dicss , filess in os.walk(os.path.join(root_Dic,value)):
            for tcgaFile in filess:
                tcgaID = re.split(pattern="_",string=tcgaFile)[0]
                if commonFileList.__contains__(tcgaID) is False:
                    os.remove(os.path.join(roots,tcgaFile))








