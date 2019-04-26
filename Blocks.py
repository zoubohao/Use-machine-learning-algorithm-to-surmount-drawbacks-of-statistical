import BasicFunction as bf
import tensorflow as tf
from tensorflow import keras

## Use the group normalization to reduce the influence of batch size

## Liner convolution trans , do not effect the traits of input.
def TransChannelsConvolutionBlock(inputTensor ,kernalSize , outChannels , blockName) :
    shapeOfInput = inputTensor.get_shape().as_list()
    C = shapeOfInput[-1]
    with tf.variable_scope(blockName):
        transFilter = bf.WeightCreation(shape=[kernalSize,kernalSize,C,outChannels],name="TransFilter")
        trans = bf.Convolution(inputTensor,transFilter,stride=[1,1,1,1],name="TransConvolution")
    return trans


def SmallUniteConvolutionBlock(inputTensor ,kernalSize , outChannels , gNum , blockName):
    assert outChannels % gNum == 0. and outChannels // gNum >= 1.
    shapeOfInput = inputTensor.get_shape().as_list()
    C = shapeOfInput[-1]
    H = shapeOfInput[1]
    W = shapeOfInput[2]
    with tf.variable_scope(blockName):
        conFilter = bf.WeightCreation(shape=[kernalSize,kernalSize,C,outChannels],name= "ConvFilter")
        conv = bf.Convolution(inputTensor,conFilter,stride=[1,1,1,1],name= "ConvolutionOp")
        conv = tf.reshape(conv, shape=[-1, H, W ,  gNum, outChannels // gNum], name="Reshape_1")
        conv = bf.BN(conv,axis=[1,2,4],name=blockName+"BNLayer")
        conv = tf.reshape(conv, shape=[-1, H, W ,outChannels], name="Reshape_2")
        finalTensor = keras.layers.PReLU(trainable=True,name=blockName + "PReluBlock")(conv)
    return  finalTensor



def DenseNetBlock(inputTensor,denseUniteNum,outputChannels,gNum,blockName):
    ## group Normalization condition
    assert outputChannels % gNum == 0. and outputChannels // gNum >= 1.
    ## attention condition
    assert outputChannels // 16. >= 1.
    shapeOfInput = inputTensor.get_shape().as_list()
    H = shapeOfInput[1]
    W = shapeOfInput[2]
    uniteBlockList = []
    with tf.variable_scope(blockName):
        copyInput = tf.identity(inputTensor)
        Conv1x1Copy = TransChannelsConvolutionBlock(copyInput,kernalSize=1,
                                                    outChannels=outputChannels,
                                                    blockName="TransCopyForFinalAdd")
        with tf.variable_scope("DenseNetBackBone"):
            ## Single convolution will not change the traits of input tensor,it is an liner trans.
            transConv = TransChannelsConvolutionBlock(inputTensor, kernalSize=1,
                                                      outChannels=outputChannels // 2,
                                                      blockName="TransInputTensorForInputToSublayer")
            uniteBlockList.append(transConv)
            for i in range(denseUniteNum):
                currentInput = tf.add_n(uniteBlockList, name="Add_" + str(i))
                with tf.variable_scope("DenseUniteBlock" + str(i)):
                    currentConv1x1 = SmallUniteConvolutionBlock(currentInput, kernalSize=1,
                                                                outChannels=outputChannels // 2, gNum=gNum,
                                                                blockName="Conv1x1_Den" + str(i))
                    currentConv3x3 = SmallUniteConvolutionBlock(currentConv1x1, kernalSize=3,
                                                                outChannels=outputChannels // 4, gNum=gNum,
                                                                blockName="Conv3x3_Den" + str(i))
                    currentConv1x1_2 = SmallUniteConvolutionBlock(currentConv3x3,kernalSize=1,gNum=gNum,
                                                                  outChannels=outputChannels // 2 ,blockName="Conv1x1_2_Den")
                    uniteBlockList.append(currentConv1x1_2)
            mediumTensor = tf.add_n(uniteBlockList, name="mediumAddAll")
            mediumTensor = SmallUniteConvolutionBlock(mediumTensor,kernalSize=1,
                                                 outChannels=outputChannels,gNum=gNum,
                                                      blockName="mediumTransUnite")
        with tf.variable_scope("AttentionBlock_Den"):
            convAttention = keras.layers.GlobalAvgPool2D(name="GlobalAvg")(mediumTensor)
            convAttention = bf.FullConnection(convAttention,units=outputChannels // 16 , blockName="Dense_1")
            convAttention = bf.FullConnection(convAttention,units=outputChannels,blockName="Dense_2")
            convAttention = tf.reshape(convAttention,shape=[-1,1,1,outputChannels],name="TransDim")
            convAttention = keras.layers.UpSampling2D(size=(H,W),
                                                      name="UpSam")(convAttention)
            convAttention = tf.nn.sigmoid(convAttention)
            attenTensor = tf.multiply(convAttention,mediumTensor,name="multiAttention")
        attenTensor = tf.add(attenTensor,mediumTensor,name="AddAttention")
        finalTensor = tf.add(Conv1x1Copy,attenTensor,name="finalAdd")
    return finalTensor


def ResNextBlock(inputTensor,parallelUnitNum,outputChannels,gNum,blockName):
    ## group Normalization condition
    assert outputChannels % gNum == 0. and outputChannels // gNum >= 1.
    ## attention condition
    assert outputChannels // 16. >= 1.
    shapeOfInput = inputTensor.get_shape().as_list()
    H = shapeOfInput[1]
    W = shapeOfInput[2]
    parallelTensorList = []
    with tf.variable_scope(blockName):
        copyInput = tf.identity(inputTensor)
        Conv1x1_Copy = TransChannelsConvolutionBlock(copyInput,kernalSize=1,
                                                     outChannels=outputChannels,
                                                     blockName="TransCopyInputForAdd")
        with tf.variable_scope("ResNextBackBone"):
            for i in range(parallelUnitNum):
                with tf.variable_scope("ParallelBlock_" + str(i)):
                    Conv1x1_1 = SmallUniteConvolutionBlock(inputTensor, kernalSize=1,
                                                           outChannels=8, gNum=gNum,
                                                           blockName="CONV1X1_Rex_1")
                    Conv3x3_4 = SmallUniteConvolutionBlock(Conv1x1_1, kernalSize=3,
                                                           outChannels=8, gNum=gNum,
                                                           blockName="CONV3X3_Rex_4")
                    Conv1x1_2 = SmallUniteConvolutionBlock(Conv3x3_4, kernalSize=1,
                                                           outChannels=outputChannels, gNum=gNum,
                                                           blockName="CONV1X1_Rex_2")
                    parallelTensorList.append(Conv1x1_2)
        mediumTensor = tf.add_n(parallelTensorList, name="mediumAdd")
        with tf.variable_scope("AttentionBlock_Rex"):
            convAttention = keras.layers.GlobalAvgPool2D(name="GlobalAvg")(mediumTensor)
            convAttention = bf.FullConnection(convAttention,units=outputChannels // 16 , blockName="Dense_1")
            convAttention = bf.FullConnection(convAttention,units=outputChannels,blockName="Dense_2")
            convAttention = tf.reshape(convAttention,shape=[-1,1,1,outputChannels],name="TransDim")
            convAttention = keras.layers.UpSampling2D(size=(H,W),
                                                      name="UpSam")(convAttention)
            convAttention = tf.nn.sigmoid(convAttention)
            attenTensor = tf.multiply(convAttention,mediumTensor,name="MultiAttention")
        attenTensor = tf.add(attenTensor,mediumTensor,name="AddAttention")
        finalTensor = tf.add(attenTensor,Conv1x1_Copy,name="finalAdd")
    return finalTensor

def RxDeThreeAddLayer(inputTensor , outputChannels , gNum, blockName ):
    with tf.variable_scope(blockName):
        with tf.variable_scope("ResNext_Units_24_Block"):
            Rex = ResNextBlock(inputTensor,parallelUnitNum=outputChannels // 32 + 6,
                               outputChannels=outputChannels,gNum=gNum,blockName="ResNext_1_Block")
            rexNetWeight = tf.Variable(1.,dtype=tf.float32,name="RexNetWeight")
            rexNetWeight = tf.nn.sigmoid(rexNetWeight)
            finalRex = tf.scalar_mul(scalar=rexNetWeight,x=Rex)
        with tf.variable_scope("DenseNet_2_Units_Blocks"):
            Den = DenseNetBlock(inputTensor,outputChannels=outputChannels,
                                denseUniteNum = 6 ,gNum=gNum,blockName="DenseNet_1_Block")
            denNetWeight = tf.Variable(1.,dtype=tf.float32,name="DenseNetWeight")
            denNetWeight = tf.nn.sigmoid(denNetWeight)
            finalDen = tf.scalar_mul(scalar=denNetWeight,x=Den)
        finalTensor = tf.add_n([finalRex,finalDen],name="FinalAddN")
    return finalTensor

## Do not change the shape of input , only change the channels .
def MimicTransformerEncoder(inputTensor,outputChannels,blockName):
    with tf.variable_scope(blockName):
        input_Copy = TransChannelsConvolutionBlock(tf.identity(inputTensor),kernalSize=1,
                                                   outChannels=outputChannels // 2,blockName="InputChannelsTrans")
        with tf.variable_scope("MultiHeadBlock"):
            with tf.variable_scope("P1_Block"):
                p1Liner = SmallUniteConvolutionBlock(inputTensor, kernalSize=1,
                                                     outChannels= outputChannels // 4, gNum=4, blockName="p1Liner")
                parallelLayer_1 = RxDeThreeAddLayer(p1Liner, outputChannels = outputChannels // 2, gNum=4,
                                                      blockName="parallelLayer_1")
            with tf.variable_scope("P2_Block"):
                p2Liner = SmallUniteConvolutionBlock(inputTensor, kernalSize=1,
                                                     outChannels=outputChannels // 4 , gNum=4, blockName="p2Liner")
                parallelLayer_2 = RxDeThreeAddLayer(p2Liner, outputChannels=outputChannels // 2, gNum=4,
                                                      blockName="parallelLayer_2")
            with tf.variable_scope("P3_Block"):
                p3Liner = SmallUniteConvolutionBlock(inputTensor, kernalSize=1,
                                                     outChannels=outputChannels // 4, gNum=4, blockName="p3Liner")
                parallelLayer_3 = RxDeThreeAddLayer(p3Liner, outputChannels=outputChannels // 2, gNum=4,
                                                      blockName="parallelLayer_3")
            concatParallelLayers_2 = tf.concat([parallelLayer_1, parallelLayer_2, parallelLayer_3],
                                               axis=-1, name="ConcatParallelLayers")
            liner_3 = SmallUniteConvolutionBlock(concatParallelLayers_2, kernalSize=1,
                                                 outChannels=outputChannels // 2, gNum=4, blockName="Non_LinerForConcat")
        with tf.variable_scope("Add_And_Nor_1"):
            mediumTensor = tf.add(liner_3, input_Copy, name="MediumADD")
            ### Layer Nor
            mediumTensor = bf.BN(mediumTensor,axis=[1,2,3],name="MediumNor")
        transMedium2Out = TransChannelsConvolutionBlock(mediumTensor, kernalSize=1,
                                                        outChannels=outputChannels, blockName="transMedium2OutChannels")
        with tf.variable_scope("FeedForward"):
            ## to simulate feedforward net .
            conv3x3_1 = SmallUniteConvolutionBlock(mediumTensor,kernalSize=3,
                                                   outChannels=outputChannels // 4,gNum=4,blockName="Conv3x3_1")
            conv2x2_2 = SmallUniteConvolutionBlock(conv3x3_1,kernalSize=2,
                                                   outChannels=outputChannels // 2,gNum=4,blockName="Conv2x2_2")
            conv1x1_3 = SmallUniteConvolutionBlock(conv2x2_2,kernalSize=1,
                                                   outChannels=outputChannels // 1,gNum=4,blockName="Conv1x1_3")
            conv1x1_T = TransChannelsConvolutionBlock(conv1x1_3,kernalSize=1,
                                                      outChannels=outputChannels,blockName="Conv1x1_T")
        with tf.variable_scope("Add_And_Nor_2"):
            finalTensor = tf.add(transMedium2Out,conv1x1_T,name="FinalADD")
            ### Layer Nor
            finalTensor = bf.BN(finalTensor,axis=[1,2,3],name="FinalNor")
        return finalTensor


def SmallResBlockWithAVG2xDownSampling(inputTensor ,kernalSize , gNum , outChannels , blockName ):
    assert outChannels % gNum == 0. and outChannels // gNum >= 1.
    with tf.variable_scope(blockName):
        res_P = SmallUniteConvolutionBlock(inputTensor,kernalSize=kernalSize,outChannels=outChannels,
                                           gNum = gNum,blockName="Res_Part_Block")
        inputTrans = TransChannelsConvolutionBlock(inputTensor,kernalSize=kernalSize,
                                                   outChannels=outChannels,
                                                   blockName="InputTensorTrans")
        m_Tensor = tf.add(res_P,inputTrans,name="Medium_Add")
        finalTensor = bf.Pooling(m_Tensor,windowShape=[2,2],poolingType="AVG",stride=[2,2],padding="VALID",
                                 name="FinalPooling")
    return finalTensor

def SmallResBlockWith2xUpSampling(inputTensor , kernelSize , gNum , outChannels , blockName):
    with tf.variable_scope(blockName):
        shapeOfInput = inputTensor.get_shape().as_list()
        B = shapeOfInput[0]
        H = shapeOfInput[1]
        W = shapeOfInput[2]
        res_P = SmallUniteConvolutionBlock(inputTensor,kernalSize=kernelSize , outChannels=outChannels,
                                           gNum=gNum,blockName="SmallResBlock")
        inputTrans = TransChannelsConvolutionBlock(inputTensor , kernalSize=kernelSize ,
                                                   outChannels=outChannels,blockName="TransInputTensor")
        m_Tensor = tf.add(res_P,inputTrans,name="mediumAdd")
        filterTensor = bf.WeightCreation(
            shape=[int(1. / outChannels * 512)  , int(1. / outChannels * 512)  ,outChannels,outChannels],
                                         name="TransposeFilter")
        finalTensor = tf.nn.conv2d_transpose(m_Tensor,filter=filterTensor,
                                             output_shape=[B , 2 * H , 2 * W , outChannels],
                                             strides=[1,2,2,1],
                                             padding="SAME",
                                             name="TransposeOp")
    return finalTensor


if __name__ == "__main__":
    testTensor = tf.placeholder(shape=[5,192,192,5],dtype=tf.float32)
    testOutput = MimicTransformerEncoder(testTensor,outputChannels=32,blockName="TestTransform")
    print(testOutput)
    tf.summary.FileWriter(logdir="d:\\test\\",graph=tf.get_default_graph())




