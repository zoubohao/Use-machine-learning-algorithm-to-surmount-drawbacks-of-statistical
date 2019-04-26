import tensorflow as tf
import numpy as np
import DCBTF
import os
import re
import math

##########
### Config
##########

batchSize = 1
featuresNum = 5
H = 192
W = 192
outLabelNum = 4
iniLR = 1e-5
epoch = 11
timesInOneEpoch = 3200
displayInforTimes = 75
modelSaveTimes = 3200
trainOrTest = "Test"
modelWeightsSavePath = "D:\ThesisModelSave\DCBTF_Model_Npy.ckpt"
haveModelParameters = True
decayRate = 0.9
decaySteps = 3200
excludeSamplesFilePath = "D:\excludeSamples.txt"
activityMappingDic = "D:\\activityMapping"
### The path is Training samples dic or testing samples dic or gross dic.
breastFilesPath = "D:\TotalSamplesNpy\BreastSamples"
liverFilePath = "D:\TotalSamplesNpy\LiverSamples"
lungFilePath = "D:\TotalSamplesNpy\LungSamples"
stomachFilePath = "D:\TotalSamplesNpy\StomachSamples"


if os.path.exists(activityMappingDic) is False:
    os.mkdir(activityMappingDic)

#########################
### Build data set .
#########################
dataFilePathList = [breastFilesPath,liverFilePath,lungFilePath,stomachFilePath]
breastSamplesSet = set()
liverSamplesSet = set()
lungSamplesSet = set()
stomachSamplesSet = set()
for i, file_path in enumerate(dataFilePathList):
    for root, dics, files in os.walk(file_path):
        for file in files:
            if i == 0:
                breastSamplesSet.add(re.split("_", file)[0])
            elif i == 1:
                liverSamplesSet.add(re.split("_", file)[0])
            elif i == 2:
                lungSamplesSet.add(re.split("_", file)[0])
            elif i == 3:
                stomachSamplesSet.add(re.split("_", file)[0])

print("There are " + str(len(breastSamplesSet)) + " breast samples for training .")
print("There are " + str(len(liverSamplesSet)) + " liver samples for training .")
print("There are " + str(len(lungSamplesSet)) + " lung samples for training .")
print("There are " + str(len(stomachSamplesSet)) + " stomach samples for training .")

breastSamplesNum = len(breastSamplesSet)
liverSamplesNum = len(liverSamplesSet)
lungSamplesNum = len(lungSamplesSet)
stomachSamplesNum = len(stomachSamplesSet)
minNum = min([breastSamplesNum , liverSamplesNum , lungSamplesNum ,
              stomachSamplesNum ])


#################
### Net Construct
#################
LRPlaceHolder = tf.placeholder(dtype=tf.float32)
with tf.variable_scope("DCBFT_Net_Construction"):
    dcbtfNet = DCBTF.DCBTF_NetWork(batchSize, featuresNum, H, W, outLabelNum)
    inputPlaceHolder = dcbtfNet.inPlaceHolder
    outputPlaceHolder = dcbtfNet.outPlaceHolder
    netOutput = dcbtfNet.Final_D_OUT
    activityMap = dcbtfNet.ActivityMap
    activityWeights = dcbtfNet.ActivityWeight


##############
### Loss Build
##############
print("Start Build Loss .")
with tf.variable_scope("Loss_Build"):
    eachBatchLoss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputPlaceHolder,
                                                               logits=netOutput,
                                                               dim=1,
                                                               name="EachBatchSoftmaxCrossEntropyLoss")
    finalFactorLabelLossMean = tf.reduce_mean(eachBatchLoss,name="FinalMeanBatchLoss")
    tf.add_to_collection("Loss",finalFactorLabelLossMean)
    finalTotalLoss = tf.add_n(tf.get_collection("Loss"),name="FinalTotalLoss")
print("Loss has been build . ")
print("Start build optimizer .")
with tf.variable_scope("OptimizerBuild"):
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optim = tf.train.MomentumOptimizer(LRPlaceHolder,momentum=0.99,
                                           use_nesterov=True,name="MomentumOptimizer")\
            .minimize(finalTotalLoss,name="MinimizerFinalTotalLoss")
print("Optimizer has been build . ")


#######################
### Build data generate
#######################
### flag infor : 0 --- breast , 1 --- liver , 2 --- lung , 3 --- stomach
def DataSetGenerate(dataSet,dataStorePath,flag):
    while True:
        for oneName in dataSet:
            copyInfor = np.load(os.path.join(dataStorePath, oneName + "_0.npy"))
            methInfor = np.load(os.path.join(dataStorePath, oneName + "_1.npy"))
            somaticInfor = np.load(os.path.join(dataStorePath, oneName + "_2.npy"))
            rnaInfor = np.load(os.path.join(dataStorePath, oneName + "_3.npy"))
            proteinInfor = np.load(os.path.join(dataStorePath, oneName + "_4.npy"))
            tumorCell = np.stack([copyInfor,methInfor,somaticInfor,rnaInfor,proteinInfor],axis=-1)
            if flag == 0 :
                outLabelData = np.array([1., 0, 0, 0], dtype=np.float32)
            elif flag == 1 :
                outLabelData = np.array([0., 1, 0, 0], dtype=np.float32)
            elif flag == 2 :
                outLabelData = np.array([0., 0, 1, 0], dtype=np.float32)
            else:
                outLabelData = np.array([0., 0, 0, 1], dtype=np.float32)
            yield np.array(tumorCell,dtype=np.float32) , outLabelData

breastDataGe = DataSetGenerate(breastSamplesSet,dataFilePathList[0],0)
liverDataGe = DataSetGenerate(liverSamplesSet,dataFilePathList[1],1)
lungDataGe = DataSetGenerate(lungSamplesSet,dataFilePathList[2],2)
stomachDataGe = DataSetGenerate(stomachSamplesSet,dataFilePathList[3],3)
dataGeList = [breastDataGe,liverDataGe,lungDataGe,stomachDataGe]
whichData = 0
saver = tf.train.Saver()
#################################
### Training and inference part .
#################################

def ReLU(mNumpy):
    resultM = np.maximum(mNumpy,0.)
    return resultM

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.)
config = tf.ConfigProto(gpu_options = gpu_options)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    if trainOrTest.lower() == "train":
        print("Start Initial Net .")
        if haveModelParameters is False:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
        else:
            saver.restore(sess,save_path=modelWeightsSavePath)
        trainingTimes = 0
        lr = iniLR
        print("Initial has completed .")
        for e in range(epoch):
            for t in range(timesInOneEpoch):
                batchInputData = []
                batchInputLabel = []
                for b in range(batchSize):
                    dataGe = dataGeList[whichData]
                    dataInput , dataLabel = dataGe.__next__()
                    batchInputData.append(dataInput)
                    batchInputLabel.append(dataLabel)
                    whichData += 1
                    if whichData == len(dataGeList) :
                        whichData = 0
                batchInputData = np.array(batchInputData)
                # print(batchInputData.shape)
                # print(batchInputData)
                batchInputLabel = np.array(batchInputLabel)
                # print(batchInputLabel.shape)
                # print(batchInputLabel)
                if trainingTimes % displayInforTimes == 0 :
                    netOutputNum = sess.run(netOutput , feed_dict={
                        inputPlaceHolder: batchInputData,
                        outputPlaceHolder: batchInputLabel,
                        LRPlaceHolder: lr
                    })
                    labelLossNum = sess.run(finalFactorLabelLossMean , feed_dict={
                        inputPlaceHolder : batchInputData,
                        outputPlaceHolder : batchInputLabel,
                        LRPlaceHolder : lr
                    })
                    totalLossNum = sess.run(finalTotalLoss , feed_dict={
                        inputPlaceHolder : batchInputData,
                        outputPlaceHolder : batchInputLabel,
                        LRPlaceHolder : lr
                    })
                    print("It is " + str(trainingTimes) + " for training .")
                    print("The learning rate is " , lr)
                    print("Label loss is " , labelLossNum)
                    print("Total loss is " , totalLossNum)
                    print("The net output is " , netOutputNum)
                    print("Label is " , batchInputLabel)
                sess.run(optim,feed_dict={
                    inputPlaceHolder : batchInputData,
                    outputPlaceHolder : batchInputLabel,
                    LRPlaceHolder : lr
                })
                if trainingTimes % decaySteps == 0 and trainingTimes != 0 :
                    lr = lr * math.pow(decayRate , trainingTimes / decaySteps + 0.0)
                if trainingTimes % modelSaveTimes == 0 and trainingTimes != 0:
                    saver.save(sess,save_path=modelWeightsSavePath)
                trainingTimes = trainingTimes + 1
    ### Test part
    else:
        print("Test part . Start initial variables .")
        saver.restore(sess,save_path=modelWeightsSavePath)
        print("Initial has been completed .")
        dataSetList = [breastSamplesSet,liverSamplesSet,lungSamplesSet,stomachSamplesSet]
        dataPathList = [breastFilesPath,liverFilePath,lungFilePath,stomachFilePath]
        resultRatioList = []
        excludeSamples = []
        np.set_printoptions(threshold=1000,suppress=True)
        activityWeightsNumpy = np.squeeze(np.array(sess.run(activityWeights)))
        print("The shape of activity weights is " , activityWeightsNumpy.shape)
        print(activityWeightsNumpy)
        for i,oneDataSet in enumerate(dataSetList):
            TNum = 0
            grossNum = 0
            ratio = 0
            storeMappingDic = os.path.join(activityMappingDic,str(i))
            if os.path.exists(storeMappingDic) is False:
                os.mkdir(storeMappingDic)
            for sampleName in oneDataSet:
                copy = np.load(os.path.join(dataPathList[i], sampleName + "_0.npy"))
                meth = np.load(os.path.join(dataPathList[i], sampleName + "_1.npy"))
                somatic = np.load(os.path.join(dataPathList[i], sampleName + "_2.npy"))
                rna = np.load(os.path.join(dataPathList[i], sampleName + "_3.npy"))
                protein = np.load(os.path.join(dataPathList[i], sampleName + "_4.npy"))
                oneTestData = np.stack([copy, meth, somatic, rna, protein], axis=-1)
                oneTestDataA = np.array(oneTestData,dtype=np.float32)
                oneTestDataF = np.reshape(oneTestDataA,newshape=[1,H,W,featuresNum])
                netOutputRes = sess.run(netOutput,feed_dict={
                    inputPlaceHolder:oneTestDataF
                })
                positionPredict = np.argmax(np.squeeze(netOutputRes))
                print("Predict : " + str(positionPredict) + " Truth :" + str(i) + " , " + str(grossNum))
                if positionPredict == i :
                    TNum += 1
                    activityMapNumpy = np.squeeze(np.array(sess.run(activityMap,feed_dict={
                        inputPlaceHolder: oneTestDataF
                    })))
                    channelsWeights = activityWeightsNumpy[:,i]
                    channelsWeights = np.squeeze(channelsWeights)
                    aImageNum = np.zeros(shape=[H,W],dtype=np.float32)
                    for k,oneWeight in enumerate(channelsWeights):
                        aImageNum = aImageNum + oneWeight * activityMapNumpy[:,:,k]
                    aImageNum = ReLU(aImageNum)
                    mappingFileName = os.path.join(storeMappingDic , sampleName + ".npy")
                    np.save(mappingFileName,aImageNum)
                else:
                    excludeSamples.append(str(sampleName) + "_" + str(i))
                grossNum += 1
            ratio = TNum / grossNum + 0.0
            resultRatioList.append(ratio)
        print("The result ratio :" + str(resultRatioList))
        print("There are " + str(len(excludeSamples)) + " samples needing to exclude .")
        with open(excludeSamplesFilePath ,mode="w") as f :
            for name in excludeSamples :
                f.write(name + "\n")



