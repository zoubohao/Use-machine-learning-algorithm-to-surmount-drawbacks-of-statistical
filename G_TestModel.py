import tensorflow as tf
import numpy as np
import DCBTF
import os
import re
import cv2


batchSize = 1
featuresNum = 5
H = 192
W = 192
outLabelNum = 4
iniLR = 1e-5
epoch = 10
timesInOneEpoch = 3200
displayInforTimes = 75
modelSaveTimes = 3200
trainOrTest = "Test"
modelWeightsSavePath = "D:\ThesisModelSave\DCBTF_Model_Npy.ckpt"
haveModelParameters = True
decayRate = 0.9
decaySteps = 3200
excludeSamplesFilePath = "D:\excludeSamples.txt"
activityMappingDic = "D:\\TestActivityMapping"
### The path is Test samples dic .
breastFilesPath = "D:\TestingSamplesNpy\BreastSamples"
liverFilePath = "D:\TestingSamplesNpy\LiverSamples"
lungFilePath = "D:\TestingSamplesNpy\LungSamples"
stomachFilePath = "D:\TestingSamplesNpy\StomachSamples"
testFilesPaths = [breastFilesPath,liverFilePath,lungFilePath,stomachFilePath]
breastSamplesSet = set()
liverSamplesSet = set()
lungSamplesSet = set()
stomachSamplesSet = set()
for i ,filesPath in enumerate(testFilesPaths):
    for root , dics , files in os.walk(filesPath) :
        for file in files:
            if i == 0 :
                breastSamplesSet.add(re.split("_", file)[0])
            elif i == 1:
                liverSamplesSet.add(re.split("_", file)[0])
            elif i == 2 :
                lungSamplesSet.add(re.split("_", file)[0])
            else:
                stomachSamplesSet.add(re.split("_", file)[0])
print("There are " + str(len(breastSamplesSet)) + " for testing in breast cancer .")
print("There are " + str(len(liverSamplesSet)) + " for testing in liver cancer .")
print("There are " + str(len(lungSamplesSet)) + " for testing in lung cancer .")
print("There are " + str(len(stomachSamplesSet)) + " for testing in stomach cancer .")
###############
##Net construct
###############
LRPlaceHolder = tf.placeholder(dtype=tf.float32)
with tf.variable_scope("DCBFT_Net_Construction"):
    dcbtfNet = DCBTF.DCBTF_NetWork(batchSize, featuresNum, H, W, outLabelNum)
    inputPlaceHolder = dcbtfNet.inPlaceHolder
    outputPlaceHolder = dcbtfNet.outPlaceHolder
    netOutput = dcbtfNet.Final_D_OUT
    activityWeight = dcbtfNet.ActivityWeight
    activityMapping = dcbtfNet.ActivityMap

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

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.)
config = tf.ConfigProto(gpu_options = gpu_options)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess :
    print("Start restore session .")
    tf.train.Saver().restore(sess,save_path=modelWeightsSavePath)
    print("Restore has completed .")
    testSamplesList = [breastSamplesSet,liverSamplesSet,lungSamplesSet,stomachSamplesSet]
    resultRatio = []
    activityWeightNumpy = np.squeeze(np.array(sess.run(activityWeight)))
    print(activityWeightNumpy)
    for i , oneClassSet in enumerate(testSamplesList) :
        TNum = 0
        grossNum = 0
        ratio = 0
        for sampleName in oneClassSet :
            print(sampleName)
            copyInfor = np.load(os.path.join(testFilesPaths[i], sampleName + "_0.npy"))
            methInfor = np.load(os.path.join(testFilesPaths[i], sampleName + "_1.npy"))
            somaticInfor = np.load(os.path.join(testFilesPaths[i], sampleName + "_2.npy"))
            rnaInfor = np.load(os.path.join(testFilesPaths[i], sampleName + "_3.npy"))
            proteinInfor = np.load(os.path.join(testFilesPaths[i], sampleName + "_4.npy"))
            tumorCell = np.stack([copyInfor,methInfor,somaticInfor,rnaInfor,proteinInfor],axis=-1)
            inputData = np.array(tumorCell)
            oneTestDataF = np.reshape(inputData, newshape=[1, H, W, featuresNum])
            netOutputRes = sess.run(netOutput, feed_dict={
                inputPlaceHolder: oneTestDataF
            })
            print("The result is ",netOutputRes)
            positionPredict = np.argmax(np.squeeze(netOutputRes))
            print("Predict : " + str(positionPredict) + " Truth :" + str(i) + " , " + str(grossNum))
            activityMapNumpy = np.squeeze(np.array(sess.run(activityMapping,feed_dict={
                inputPlaceHolder:oneTestDataF
            })))
            print("activity mapping shape ",activityMapNumpy.shape)
            channelsWeights = activityWeightNumpy[:,i]
            channelsWeights = np.squeeze(channelsWeights)
            print(channelsWeights)
            print(channelsWeights.shape)
            imageFocusOn = np.zeros(shape=[H,W],dtype=np.float32)
            for k,weight in enumerate(channelsWeights) :
                imageFocusOn = imageFocusOn + weight * activityMapNumpy[:,:,k] + 0.0
            if positionPredict == i :
                TNum = TNum + 1
            grossNum = grossNum + 1
            cv2.imwrite(os.path.join(activityMappingDic,sampleName + str(i) + ".png"),imageFocusOn)
        print(float(TNum) / float(grossNum))
        resultRatio.append(float(TNum) / float(grossNum))
    print("Final ratio is ",resultRatio)








