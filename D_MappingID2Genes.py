import os
import re
import gc

if __name__ == "__main__":
    mappingFilePath = "D:\ThesisDataDownLoad\StomachCancer" \
                      "\\Stomach Cancer Somatic Mutation Non-Silent Mapping ID 2 Genes"
    fileFolderPath = "D:\ThesisDataDownLoad\StomachCancer\SomaticMutation"
    id2Genes = {}
    with open(mappingFilePath,mode="r") as f :
        for i , line in enumerate(f):
            if i != 0 :
                lineSplit = re.split(pattern="\t",string=line)
                ### 如果ID没有对应的基因，则这个基因为None
                if lineSplit[1] == "":
                    id2Genes[lineSplit[0]] = "None"
                else:
                    id2Genes[lineSplit[0]] = lineSplit[1]
    print(len(id2Genes.keys()))
    for root , dics , files in os.walk(fileFolderPath):
        ### read files in folder
        for k,file in enumerate(files):
            print(str(k) + "  " +file)
            genes2value = {}
            ### read contains of file
            with open(os.path.join(root, file), mode="r") as f:
                for j, line in enumerate(f):
                    lineSplit = re.split(pattern="\t", string=line.strip("\n"))
                    idString = lineSplit[0]
                    ### 如果该id对应的值为NA,则设置为0.
                    if lineSplit[1] == "NA" or lineSplit[1] == "":
                        idValue = float(0.0)
                    else:
                        idValue = float(lineSplit[1])
                    ### 找出对应的多个基因
                    corrGenesString = id2Genes.get(idString)
                    ### 如果该id没有对应的基因，则不处理，丢弃
                    if corrGenesString is not None:
                        corrGenes = re.split(pattern=",", string=corrGenesString)
                        for gene in corrGenes:
                            if genes2value.__contains__(gene) is False:
                                genes2value[gene] = idValue
                            else:
                                oriValue = genes2value.get(gene)
                                newValue = oriValue + idValue
                                genes2value[gene] = newValue
            with open(os.path.join(root, file + "New"), mode="w") as f:
                for key, value in genes2value.items():
                    if key != "None":
                        f.write(key + "\t" + str(value) + "\n")
            del genes2value
            if k % 50 == 0:
                gc.collect()



