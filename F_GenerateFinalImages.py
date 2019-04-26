import numpy as np
import os
import re



if __name__ == "__main__":
    root_path = "D:\ThesisDataDownLoad\\StomachCancer"
    outImagePath = "D:\TrainingSamplesNpy\\StomachSamples"
    width = 192
    height = 192
    #########
    copy_part_path = "CopyNumberData\TrainingSamples"
    methy_part_path = "Methylation\TrainingSamples"
    exon_part_path = "ExonExpression\TrainingSamples"
    RNA_part_path = "RNA-Seq\TrainingSamples"
    soma_part_path = "SomaticMutation\TrainingSamples"
    path_list = [os.path.join(root_path,copy_part_path),
                 os.path.join(root_path,methy_part_path),
                 os.path.join(root_path,soma_part_path),
                 os.path.join(root_path,RNA_part_path),
                 os.path.join(root_path,exon_part_path)]
    samplesName = []
    for root , dics , files in os.walk(path_list[1]):
        for file in files:
            sampleName = re.split("_",file)[0]
            samplesName.append(sampleName)
    print(len(samplesName))

    if os.path.exists(outImagePath) is False:
        os.mkdir(outImagePath)

    def MinMaxNorm(array):
        array = np.array(array)
        maxNum = np.max(array)
        meanNum = np.mean(array)
        if maxNum > abs(meanNum * 10.):
            for k , v in enumerate(array):
                if v >= abs(meanNum * 10.):
                    array[k] = abs(meanNum * 10.)
        maxNum = np.max(array)
        minNum = np.min(array)
        return (array - minNum) / (maxNum - minNum) + 0.

    for name in samplesName:
        print(name)
        for i,path_dic in enumerate(path_list):
            valuesList = []
            with open(os.path.join(path_dic, name), mode="r") as fh:
                for line in fh:
                    newLine = line.strip("\n")
                    value = float(re.split("\t", newLine)[1])
                    valuesList.append(value)
            valueListNorm = MinMaxNorm(valuesList)
            valueImage = np.reshape(valueListNorm, newshape=[height, width])
            np.save(os.path.join(outImagePath, name + "_" + str(i) + ".npy"), valueImage)
            # if os.path.exists(os.path.join(path_dic, name)) is True :
            #     with open(os.path.join(path_dic, name), mode="r") as fh:
            #         for line in fh:
            #             newLine = line.strip("\n")
            #             value = float(re.split("\t", newLine)[1])
            #             valuesList.append(value)
            #     valueListNorm = MinMaxNorm(valuesList)
            #     valueImage = np.reshape(valueListNorm, newshape=[height, width])
            #     np.save(os.path.join(outImagePath, name + "_" + str(i) + ".npy"), valueImage)
            # else:
            #     np.save(os.path.join(outImagePath, name + "_" + str(i) + ".npy"),
            #               np.zeros(shape=[192,192],dtype=np.float32))









