import numpy as np
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# 定义用来存放图像数据 图像标签 图像名称list  最后返回的cifar_image cifar_label即是图像cifar-10 对应的数据和标签
def CreatData():
    # 创建训练样本
    # 依次加载batch_data_i,并合并到x,y
    x = []
    y = []
    for i in range(1, 6):
        batch_path = 'cifar-10-batches-py\data_batch_%d' % (i)
        batch_dict = unpickle(batch_path)

        train_batch = batch_dict[b'data'].astype('float')
        train_labels = np.array(batch_dict[b'labels'])
        x.append(train_batch)
        y.append(train_labels)
    # 将5个训练样本batch合并为50000x3072，标签合并为50000x1
    # np.concatenate默认axis=0，为纵向连接
    traindata = np.concatenate(x)
    trainlabels = np.concatenate(y)

    # 创建测试样本
    # 直接写cifar-10-batches-py\test_batch会报错，因此把/t当作制表符了，应用\\;
    #    test_dict=unpickle("cifar-10-batches-py\\test_batch")

    # 建议使用os.path.join()函数
    testpath = os.path.join('cifar-10-batches-py', 'test_batch')
    test_dict = unpickle(testpath)
    testdata = test_dict[b'data'].astype('float')
    testlabels = np.array(test_dict[b'labels'])
    traindata = traindata.reshape(50000,3,32,32).transpose(0,2,3,1)
    testdata = testdata.reshape(10000,3,32,32).transpose(0,2,3,1)
    return traindata, trainlabels, testdata, testlabels