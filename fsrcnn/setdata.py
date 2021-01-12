import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import cifar10
def setdata():
    traindata, trainlabels, testdata, testlabels =cifar10.CreatData()
    # print(train_X[0])
    # train_X = train_X.reshape(60000,28,28,1)
    # test_X  = test_X.reshape(10000,28,28,1)
    print(traindata.shape, trainlabels.shape, testdata.shape, testlabels.shape)
    traindata = traindata/255.0
    testdata = testdata/255.0
    # plt.imshow(traindata[30])
    train_X = tf.image.resize(traindata[0:25000], (56, 56), method='bicubic')
    train_X = tf.image.resize(train_X, (28, 28), method='bicubic')
    testdata =tf.image.resize(testdata, (56, 56), method='bicubic')
    testdata =tf.image.resize(testdata, (28, 28), method='bicubic')
    trainlabels = trainlabels[0:25000]
    # show picture
    # plt.imshow(traindata[30])
    # plt.show()
    return train_X, trainlabels, testdata, testlabels

# test = setdata()