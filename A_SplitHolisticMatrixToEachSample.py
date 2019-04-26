import re
import os

### CopyNumberData ExonExpression Methylation RNA-Seq SomaticMutation
### CopyNumberData ExonExpressionData Methylation450kData GenesExpressionData SomaticMutationData


if __name__ == "__main__":
    file_Dic = "D:\ThesisDataDownLoad\StomachCancer"
    file_name = "Stomach Cancer Somatic Mutation Non-Silent Data"
    data_type = "SomaticMutationData"
    mode = "r"
    file_Output_Dic = os.path.join(file_Dic,"SomaticMutation")
    fileHandles = []
    if os.path.exists(file_Output_Dic) is False:
        os.mkdir(file_Output_Dic)
    with open(os.path.join(file_Dic,file_name),mode=mode) as f :
        for i , line in enumerate(f):
            print(i)
            if i == 0 :
                samples = re.split(pattern="\t",string=line.strip("\n"))
                for j , sampleID in enumerate(samples):
                    if j != 0 :
                        fileHandle = open(os.path.join(file_Output_Dic,sampleID + "_" + data_type),
                                          mode="w")
                        fileHandles.append(fileHandle)
            else:
                dataS = re.split(pattern="\t",string=line.strip("\n"))
                for j , data in enumerate(dataS):
                    if j != 0 :
                        fileHandles[j-1].write(dataS[0] + "\t" +  data + "\n")
        for file_handle in fileHandles:
            file_handle.close()









