import tensorflow as tf
import math
from tensorflow import keras
import numpy as np

c = 5e-4

def WeightCreation(shape ,ifAddRe = True , name = None):
    weight = tf.Variable(initial_value=
                         tf.truncated_normal(shape=shape,dtype=tf.float32,
                                             stddev=math.pow(1.25 / np.sum(shape),0.5)),
                         name=name)
    if ifAddRe:
        l2Loss = tf.nn.l2_loss(weight)
        l2Loss = tf.multiply(c,l2Loss)
        tf.add_to_collection("Loss",l2Loss)
    return weight


def BN(inputTensor , axis = -1 , ep = 1e-6,name = None):
    return keras.layers.BatchNormalization(axis=axis,epsilon=ep,name=name)(inputTensor)


def Convolution(inputTensor , filWeight , stride ,dataFormat = "NHWC"  ,padding = "SAME",name = None):
    return tf.nn.conv2d(input=inputTensor,
                        filter=filWeight,
                        strides=stride,
                        data_format=dataFormat,
                        padding=padding,
                        name=name)


def Pooling (inputTensor , windowShape ,poolingType , stride, dataFormat = "NHWC" ,padding = "SAME",name = None):
    return tf.nn.pool(input=inputTensor,
                      window_shape=windowShape,
                      pooling_type=poolingType,
                      strides=stride,
                      data_format=dataFormat,
                      padding=padding,
                      name=name)


def Dropout(inputTensor , keepPro = 0.7,name = None):
    return keras.layers.Dropout(rate=keepPro,name=name)(inputTensor)


def BiasCreation(units , name):
    initialValue = np.zeros(shape=[units],dtype=np.float32)
    bias = tf.Variable(initial_value=initialValue,dtype=tf.float32,name=name)
    l2Bias = tf.multiply(c,tf.nn.l2_loss(t=bias,name=name + "BiasL2Loss"))
    tf.add_to_collection("Loss",l2Bias)
    return bias


def FullConnection(inputTensor,units,blockName):
    shapeOfInput = inputTensor.get_shape().as_list()
    inUnits = shapeOfInput[1]
    with tf.variable_scope(blockName):
        weights = WeightCreation(shape=[inUnits, units], name="Weight")
        biasIni = np.zeros(shape=[units],dtype=np.float32)
        bias = tf.Variable(initial_value=biasIni,dtype=tf.float32)
        l2Loss = tf.nn.l2_loss(bias,name="BiasL2Loss")
        l2Loss = tf.multiply(c,l2Loss)
        tf.add_to_collection("Loss",l2Loss)
        layer = tf.add(tf.matmul(inputTensor,weights,name="MatmulTrans"),bias,name="AddBias")
        layer = BN(layer,axis=1,name=blockName + "BNLayer")
        layer = keras.layers.PReLU(trainable=True,name=blockName + "PRELU")(layer)
        layer = Dropout(layer,name="Dropout")
    return layer


def FullConnectionWithoutTrans(inputTensor , units , blockName):
    shapeOfInput = inputTensor.get_shape().as_list()
    inUnits = shapeOfInput[1]
    with tf.variable_scope(blockName):
        weights = WeightCreation(shape=[inUnits,units],name="Weight")
        biasIni = np.zeros(shape=[units],dtype=np.float32)
        bias = tf.Variable(initial_value=biasIni,dtype=tf.float32)
        l2Loss = tf.nn.l2_loss(bias,name="BiasL2Loss")
        l2Loss = tf.multiply(c,l2Loss)
        tf.add_to_collection("Loss",l2Loss)
        finalTensor = tf.add(tf.matmul(inputTensor,weights,name="MatmulTrans"),bias,name="AddBias")
    return finalTensor







