import tensorflow as tf
import math
from tensorflow import keras
import numpy as np
import re
import matplotlib.pyplot as plt

# data_type_map = {"CopyNumberData": "CopyNumberData",
#                  "ExonExpressionData": "ExonExpression",
#                  "GenesExpressionData": "RNA-Seq",
#                  "Methylation450kData": "Methylation",
#                  "SomaticMutationData": "SomaticMutation"}
# for key , value in data_type_map.items():
#     print(key)
#     print(value)
#
# np.stack([],axis=1)

# sampleNumPlaceHolder = tf.placeholder(dtype=tf.float32,shape=[3,5])
# labelPlaceHolder = tf.placeholder(dtype=tf.float32,shape=[3,5])
# transConstanst = tf.constant(value=1.,dtype=tf.float32,shape=[5,1])
# multi = tf.multiply(sampleNumPlaceHolder,labelPlaceHolder)
# finalMatrix = tf.matmul(multi,transConstanst)
# finalMatrix = tf.squeeze(finalMatrix)
# a = tf.get_variable("a",shape=[4,5,6],dtype=tf.float32,initializer=tf.truncated_normal_initializer())
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     num = sess.run(finalMatrix , feed_dict={
#         sampleNumPlaceHolder:[[600,152,320,210,500],
#                               [600, 152, 320, 210, 500],
#                               [600, 152, 320, 210, 500]],
#         labelPlaceHolder : [[0 , 0 , 0 , 1 , 0],
#                             [0, 1, 0, 0, 0],
#                             [0, 0, 0, 0, 1]]
#     })
#     print(num)
#     tf.train.Saver().save(sess,save_path="d:\ThesisModelSave\\test.ckpt")
#
# import re
# import skimage.io as io
# values = []
# with open("D:\ThesisDataDownLoad\BreastCancer\CopyNumberData\TrainingSamples\TCGA-A1-A0SH-01",
#           mode="r") as f:
#     for line in f:
#         newLine = line.strip("\n")
#         values.append(float(re.split("\t",newLine)[1]))
# print(np.max(values))
# print(np.min(values))
# print(np.mean(values))
# k = 0
# meanNum = np.mean(values)
# for v in values:
#     if v >=  abs(10. * meanNum):
#         k = k + 1
# print(k)
# import  cv2
# np.set_printoptions(threshold=np.inf)
# imageArray = np.array(cv2.imread("D:\TrainingSamples\BreastCancer\TCGA-A1-A0SB-01_1.jpg",cv2.IMREAD_GRAYSCALE))
# print(imageArray.shape)
# print(imageArray)
# import re
# a = re.match(pattern=".*(\\.npy)$",string="FinalActivationGenesList.txt")
# print(a)



# import sklearn.cluster.k_means_ as KMEANS
# path = "D:\ActivityMapping\\0\TCGA-A2-A0CT-01.npy"
# kmeans = KMEANS.KMeans(n_clusters=3,max_iter=15000,n_init=346)
# data = np.load(path)
# data = np.reshape(data,newshape=[-1,1])
# data = np.squeeze(data)
# data = np.reshape(data,newshape=[-1,1])
# kmeans.fit(data)
# shapeOfData = data.shape
# x = [i for i in range(shapeOfData[0])]
# y = [j for j in data]
# print(kmeans.cluster_centers_)
# plt.scatter(x,y,marker=".",s=1,c=kmeans.labels_)
# plt.xlabel("The Flatten Position Of Each Gene")
# plt.ylabel("The CAM Value Of Each Gene")
# plt.title("The Result Of Sample TCGA-A2-A0CT-01 With K-Means Of Three Clusters")
# plt.show()
#
#
#
# path = "D:\ActivityMapping\\0\TCGA-AN-A0XL-01.npy"
# data = np.load(path)
# data = np.reshape(data,newshape=[-1,1])
# data = np.squeeze(data)
# plt.hist(x=data,bins=300,edgecolor="black")
# plt.xlabel("The Value Of Genes In CAM Image")
# plt.ylabel("Genes Quantity Frequency")
# plt.title("CAM Value Frequency Hist Figure Of Sample TCGA-AN-A0XL-01 ")
# plt.show()


import re
dataList = []
with open("d:\\StatisticResult_0.txt",mode="r") as fh :
    for line in fh:
        oneLine = line.strip("\n")
        dataList.append(int(re.split("\t",oneLine)[1]))
plt.hist(dataList,bins=300,edgecolor="black")
plt.title("The Frequency Distribution Of Clustering Score Of Breast Cancer")
plt.xlabel("The Score Of Clustering")
plt.ylabel("The Frequency Of Each Score")
plt.show()



# filePath = "D:\ThesisDataDownLoad\BreastCancer\Breast Cancer Gene Expression RNA-Seq Data Log trans"
# sampleStorePath = "D:\ThesisDataDownLoad\BreastCancer\samplesFlag.txt"
# samplesList = None
# with open(filePath,mode="r") as fh:
#     for line in fh:
#         oneline = line.strip("\n")
#         samplesList = re.split("\t",oneline)
#         break
# with open(sampleStorePath,mode="w") as fh:
#     for sampleName in samplesList:
#         if sampleName != "sample"  :
#             partList = re.split("-", sampleName)
#             print(partList)
#             flag = int(partList[-1])
#             if flag >= 10:
#                 fh.write(sampleName + "\t" + "N" + "\n")
#             else:
#                 fh.write(sampleName + "\t" + "T" + "\n")


# regex = "^TCGA-.*"
# string = "TCGA-10-10"
# regexO = re.compile(regex)
# print(regexO.findall(string))

# import scipy.stats as stats
#
# x = np.random.normal(loc=1.,scale=2.,size=[512])
# y = np.random.normal(loc=1.,scale=2.,size=[512])
# print(stats.shapiro(x))
# print(stats.shapiro(x)[1])
# print(stats.ttest_ind(x,y))
# print(stats.ttest_ind(x,y)[1])
#
# import scipy.stats as stats
# epoch = 500
# times = 1000
# finalAResult = []
# for j in range(epoch):
#     k = 0
#     print(j)
#     for i in range(times):
#         x = np.random.normal(loc=0., scale=2., size=1104)
#         y = np.random.normal(loc=0., scale=2., size=10)
#         pValue = stats.ttest_ind(x, y, equal_var=True)[1]
#         if pValue <= 0.05:
#             k += 1
#     finalAResult.append(k)
# finalBResult = []
# for j in range(epoch):
#     k = 0
#     print(j)
#     for i in range(times):
#         x = np.random.normal(loc=0., scale=2., size=1104)
#         y = np.random.normal(loc=0., scale=2., size=1104)
#         pValue = stats.ttest_ind(x, y, equal_var=True)[1]
#         if pValue <= 0.05:
#             k += 1
#     finalBResult.append(k)
# print(stats.ranksums(finalAResult,finalBResult))

# filePath = "d:\\goa_human.gaf"
# geneNamesSet = set()
# with open(filePath , mode="r") as fh :
#     for line in fh:
#         oneLine = line.strip("\n")
#         geneName =  re.split("\t",oneLine)[2]
#         geneNamesSet.add(geneName)
#
# with open("d:\\WholeGenesSet.txt" ,mode="w") as fh :
#     for geneName in geneNamesSet:
#         fh.write(geneName + "\n")





# filePath = "d:\\limma_genes_breast_test.csv"
# outputPath = "d:\\limmaBreastGenes.txt"
# with open(filePath,mode="r") as fh :
#     with open(outputPath,mode="w") as wh :
#         for i, line in enumerate(fh):
#             if i != 0 :
#                 oneLine = line.strip("\n")
#                 inforList = re.split(",", oneLine)
#                 if float(inforList[4]) <= 0.01:
#                     wh.write(inforList[0][1:-1] + "\n")


# filePath = "d:\\gene2go"
# outFile = "d:\\id2symbol"
# geneIdSet = set()
# with open(filePath,mode="r") as fh :
#     for line in fh:
#         oneLine = line.strip("\n")
#         inforList = re.split("\t", oneLine)
#         geneId = inforList[1]
#         print(geneId)
#         geneIdSet.add(geneId)
# with open(outFile,mode="w") as wh :
#     for geneId in geneIdSet:
#         wh.write(geneId + "\n")


# filePath = "d:\\CancerGenesOverAll.txt"
# outputPath = "d:\\CancerGenesSet.txt"
# genesSet = set()
# geneRex = ".{2,8}"
# objectRex = re.compile(geneRex)
# objectRex.match()
# with open(filePath,mode="r") as fh :
#     for line in fh:
#         oneLine = line.strip("\n")
#         inforList = re.split(";", oneLine)
#         print("infor list is " ,inforList)
#         geneNames = inforList[1:-1]
#         print(geneNames)
#         for geneName in geneNames:
#             print(geneName)
#             genesSet.add(geneName)
#
# with open(outputPath,mode="w") as wh :
#     for geneName in genesSet:
#         wh.write(geneName + "\n")


# goFile = "d:\\goa_human.gaf"
# inputFile = "d:\\N_StomachStrongRelatedGenesList.txt"
# outputFile = "d:\\N_StomachModifyStrongRelated.txt"
# allGenesSet = set()
# with open(goFile , mode="r") as fh :
#     for line in fh:
#         oneLine = line.strip("\n")
#         inforList = re.split("\t",oneLine)
#         geneName = inforList[2]
#         allGenesSet.add(geneName)
# with open(inputFile,mode="r") as fh :
#     with open(outputFile,mode="w") as oh :
#         for line in fh :
#             oneLine = line.strip("\n")
#             if oneLine in allGenesSet :
#                 oh.write(oneLine + "\n")


# x = np.random.normal(0.,2.,[2,4])
# print(x)
# y = np.maximum(x,0.)
# print(y)


