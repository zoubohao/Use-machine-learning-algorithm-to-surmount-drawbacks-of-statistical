import BasicFunction as bf
import  Blocks as bs
import tensorflow as tf
from tensorflow import keras
import math
import numpy as np


## This net name is
## Diversity Convolution Units Blocks To Simulate Transformer With FPN NetWork !!! DCBTF Net !!!

class DCBTF_NetWork :

    def __init__(self,batchSize , inChannels , H , W , outLabelNum):
        print("!!!!!! WARNING !!!!!!")
        print("!!! The height and width of the input must be the integer times of 32 !!!")
        print("!!!!!! WARNING !!!!!!")
        assert H % 32. == 0 and W % 32. == 0
        self.__outLabelNum = outLabelNum
        self.inPlaceHolder = tf.placeholder(shape=[batchSize,H,W,inChannels],dtype=tf.float32)
        self.outPlaceHolder = tf.placeholder(shape=[batchSize,outLabelNum],dtype=tf.float32)
        self.C1 = None
        self.C2 = None
        self.C3 = None
        self.C4 = None
        self.C5 = None
        self.P5 = None
        self.P4 = None
        self.P3 = None
        self.P2 = None
        self.P2_D_OUT = None
        self.P3_D_OUT = None
        self.P4_D_OUT = None
        self.P5_D_OUT = None
        self.ActivityMap = None
        self.ActivityWeight = None
        self.Final_D_OUT = None
        print("Start Encoder net build .")
        meTensor = self.__Encoder_Net_Build()
        print("Encoder net has been build .")
        print("Start decoder net build .")
        self.__Decoder_Net_Build(meTensor)
        print("Decoder net has been build .")

    def __Encoder_Net_Build(self):
        with tf.variable_scope("ENCODER"):

            ######################
            ### Bottom to Up
            ######################

            ### 32 channels
            with tf.variable_scope("C1_Block"):
                C1_M = bs.SmallUniteConvolutionBlock(self.inPlaceHolder, kernalSize=8,
                                                     outChannels=16, gNum=4, blockName="Initial_Conv6x6_C1")
                C1_M = bs.MimicTransformerEncoder(C1_M,outputChannels=32,blockName="Transformer_0")
                self.C1 = C1 = bf.Pooling(C1_M, windowShape=[2, 2], poolingType="AVG", stride=[2, 2],
                                padding="VALID", name="DownSampling_1")
                print("C1 ",C1)
            ### 64 channels
            with tf.variable_scope("C2_Block"):
                C2_M = bs.SmallUniteConvolutionBlock(C1,kernalSize=4,
                                                     outChannels=32,gNum=4,blockName="Initial_Conv5x5_C2")
                C2_M = bs.MimicTransformerEncoder(C2_M, outputChannels=64, blockName="Transformer_1")
                self.C2 = C2 = bf.Pooling(C2_M, windowShape=[2, 2], poolingType="MAX", stride=[2, 2],
                                padding="VALID", name="DownSampling_2")
                print("C2 ",C2)
            ### 128 channels
            with tf.variable_scope("C3_Block"):
                C3_M = bs.SmallUniteConvolutionBlock(C2,kernalSize=4,
                                                     outChannels=64,gNum=4,blockName="Initial_Conv4x4_C3")
                C3_M = bs.MimicTransformerEncoder(C3_M, outputChannels=128, blockName="Transformer_2")
                self.C3 = C3 = bf.Pooling(C3_M, windowShape=[2, 2], poolingType="MAX", stride=[2, 2],
                                padding="VALID", name="DownSampling_3")
                print("C3 ",C3)
            ### 256 channels
            with tf.variable_scope("C4_Block"):
                C4_M = bs.SmallUniteConvolutionBlock(C3,kernalSize=2,
                                                     outChannels=128,gNum=4,blockName="Initial_Conv3x3_C4")
                C4_M = bs.MimicTransformerEncoder(C4_M, outputChannels=256, blockName="Transformer_3")
                self.C4 = C4 = bf.Pooling(C4_M, windowShape=[2, 2], poolingType="MAX", stride=[2, 2],
                                padding="VALID", name="DownSampling_4")
                print("C4 ",C4)
            ### 512 channels
            with tf.variable_scope("C5_Block"):
                C5_M = bs.SmallUniteConvolutionBlock(C4,kernalSize=1,
                                                     outChannels=256,gNum=4,blockName="Initial_Conv2x2_C5")
                C5_M = bs.MimicTransformerEncoder(C5_M, outputChannels=512, blockName="Transformer_5")
                self.C5 = C5 = bf.Pooling(C5_M, windowShape=[2, 2], poolingType="MAX", stride=[2, 2],
                                padding="VALID", name="DownSampling_5")
                print("C5 ",C5)


            #####################
            ### Up to Bottom
            #####################

            ### 512
            with tf.variable_scope("P5_Block"):
                self.P5 = P5_Out = bs.SmallUniteConvolutionBlock(C5, kernalSize=1,
                                                       outChannels=512, gNum=4, blockName="P5_Out")
                print("P5 ",P5_Out)

            ### 256
            with tf.variable_scope("P4_Block"):
                P5_Trans = bs.TransChannelsConvolutionBlock(P5_Out, kernalSize=1, outChannels=256
                                                            , blockName="P5_Trans")
                P5_Up = keras.layers.UpSampling2D(size=(2, 2),
                                                  name="UpSam_P5")(P5_Trans)
                P4_M = bs.SmallUniteConvolutionBlock(C4, kernalSize=1, outChannels=256, gNum=4,
                                                     blockName="P4_M")
                self.P4 = P4_Out = tf.add(P4_M, P5_Up, name="P4_Out")
                print("P4 ",P4_Out)

            ### 128
            with tf.variable_scope("P3_Block"):
                P4_Trans = bs.TransChannelsConvolutionBlock(P4_Out, kernalSize=1, outChannels=128,
                                                            blockName="P4_Trans")
                P4_Up = keras.layers.UpSampling2D(size=(2, 2),
                                                  name="UpSam_P4")(P4_Trans)
                P3_M = bs.SmallUniteConvolutionBlock(C3, kernalSize=1, outChannels=128, gNum=4,
                                                     blockName="P3_M")
                self.P3 = P3_Out = tf.add(P4_Up, P3_M, name="P3_Out")
                print("P3 ",P3_Out)

            ### 64
            with tf.variable_scope("P2_Block"):
                P3_Trans = bs.TransChannelsConvolutionBlock(P3_Out, kernalSize=1, outChannels=64,
                                                            blockName="P3_Trans")
                P3_Up = keras.layers.UpSampling2D(size=(2, 2),
                                                  name="UpSam_P3")(P3_Trans)
                P2_M = bs.SmallUniteConvolutionBlock(C2, kernalSize=1, outChannels=64, gNum=4,
                                                     blockName="P2_M")
                self.P2 = P2_Out = tf.add(P3_Up, P2_M, name="P2_Out")
                print("P2 ",P2_Out)


            ##########################
            ### sub process
            ##########################
            with tf.variable_scope("P2_D_Block"):
                P2_D = bs.SmallResBlockWithAVG2xDownSampling(self.P2,kernalSize=4,gNum=4,
                                                                   outChannels=64,blockName="P2_D_C1")
                P2_D = bs.SmallResBlockWithAVG2xDownSampling(P2_D,kernalSize=3,gNum=4,
                                                                   outChannels=128,blockName="P2_D_C2")
                P2_D = bs.SmallResBlockWithAVG2xDownSampling(P2_D,kernalSize=2,gNum=4,
                                                                   outChannels=256,blockName="P2_D_C3")
                self.P2_D_OUT  = bs.SmallUniteConvolutionBlock(P2_D, kernalSize=1,
                                                     outChannels=512, gNum=4, blockName="P2_D_C4")
            with tf.variable_scope("P3_D_Block"):
                P3_D = bs.SmallResBlockWithAVG2xDownSampling(self.P3,kernalSize=3,gNum=4,
                                                                   outChannels=128,blockName="P3_D_C1")
                P3_D = bs.SmallResBlockWithAVG2xDownSampling(P3_D,kernalSize=2,gNum=4,
                                                                   outChannels=256,blockName="P3_D_C2")
                self.P3_D_OUT  = bs.SmallUniteConvolutionBlock(P3_D,kernalSize=1,outChannels=512,
                                                     gNum=4,blockName="P3_D_C3")
            with tf.variable_scope("P4_D_Block"):
                P4_D = bs.SmallResBlockWithAVG2xDownSampling(self.P4,kernalSize=2,gNum=4,
                                                                   outChannels=256,blockName="P4_D_C1")
                self.P4_D_OUT = bs.SmallUniteConvolutionBlock(P4_D,kernalSize=1,outChannels=512,gNum=4,
                                                     blockName="P4_D_C2")
            with tf.variable_scope("P5_D_Block"):
                self.P5_D_OUT = bs.SmallUniteConvolutionBlock(self.P5,kernalSize=1,outChannels=512,gNum=4,
                                                     blockName="P5_D_C1")
            with tf.variable_scope("ImagesAdd"):
                ###2
                p2_weight = tf.Variable(initial_value=1.,dtype=tf.float32,name="p2_weight")
                p2_trans = tf.nn.sigmoid(p2_weight)
                p2_weight_out = tf.scalar_mul(scalar=p2_trans,x=self.P2_D_OUT)
                ###3
                p3_weight = tf.Variable(initial_value=1.,dtype=tf.float32,name="p3_weight")
                p3_trans = tf.nn.sigmoid(p3_weight)
                p3_weight_out = tf.scalar_mul(scalar=p3_trans,x=self.P3_D_OUT)
                ###4
                p4_weight = tf.Variable(initial_value=1.,dtype=tf.float32,name="p4_weight")
                p4_trans = tf.nn.sigmoid(p4_weight)
                p4_weight_out = tf.scalar_mul(scalar=p4_trans,x=self.P4_D_OUT)
                ###5
                p5_weight = tf.Variable(initial_value=1.,dtype=tf.float32,name="p5_weight")
                p5_trans = tf.nn.sigmoid(p5_weight)
                p5_weight_out = tf.scalar_mul(scalar=p5_trans,x=self.P5_D_OUT)
                mediumActiveMap = tf.concat(values=[p2_weight_out,p3_weight_out,p4_weight_out,p5_weight_out],axis=-1
                                            ,name = "FinalConcatInfor")
        return mediumActiveMap


    def __Decoder_Net_Build(self,mediumActiveMap):
        with tf.variable_scope("DECODER"):
                mediumActiveMap_2 = bs.SmallResBlockWith2xUpSampling(mediumActiveMap,kernelSize=1,gNum=4,outChannels=512,
                                                                     blockName="ActiveMap2xUp")
                mediumActiveMap_4 = bs.SmallResBlockWith2xUpSampling(mediumActiveMap_2,kernelSize=2,gNum=4,outChannels=256,
                                                                     blockName="ActiveMap4xUp")
                mediumActiveMap_8 = bs.SmallResBlockWith2xUpSampling(mediumActiveMap_4,kernelSize=3,gNum=4,outChannels=128,
                                                                     blockName="ActiveMap8xUp")
                mediumActiveMap_16 = bs.SmallResBlockWith2xUpSampling(mediumActiveMap_8,kernelSize=4,gNum=4,outChannels=64,
                                                                     blockName="ActiveMap16xUp")
                self.ActivityMap = bs.SmallResBlockWith2xUpSampling(mediumActiveMap_16,kernelSize=5,gNum=4,outChannels=32,
                                                                     blockName="ActiveMap32xUp")
                print("Activity Map :",self.ActivityMap)
                globalPoolingTensor = keras.layers.GlobalAveragePooling2D(name="FinalGlobalAVGPooling")(self.ActivityMap)
                self.ActivityWeight = tf.get_variable(name="ActivityWeights" , shape=[32 , self.__outLabelNum],
                                                      dtype=tf.float32,
                                                      initializer=tf.truncated_normal_initializer(
                                                          stddev=math.pow(1.25 / np.sum([32 , self.__outLabelNum]),0.5))
                                                      ,trainable=True)
                self.Final_D_OUT = tf.matmul(globalPoolingTensor,self.ActivityWeight,name="FinalOp")
                print("Final Output :",self.Final_D_OUT)


if __name__ == "__main__":
    ### !!!!!! WARNING !!!!!!
    ### !!! The height and width of the input must be the integer times of 32 !!!
    ### !!!!!! WARNING !!!!!!
    dcbtf = DCBTF_NetWork(batchSize=4,inChannels=5,H=192,W=192,outLabelNum=4)
    tf.summary.FileWriter("d:\\test\\",graph=tf.get_default_graph())




