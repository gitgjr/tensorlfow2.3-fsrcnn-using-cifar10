import setdata
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# hyp
epoch = 1
retrain = 0
class SRCNN():
    def __init__(self):
        model = models.Sequential()
        model.add(layers.Conv2D(64,(9,9),activation='relu',input_shape=(32,32,3),padding='valid',use_bias='True',))
        model.add(layers.Conv2D(32,(1,1), activation='relu', padding='valid', use_bias=True))
        model.add(layers.Conv2D(5,(3,3), activation='relu', padding='same', use_bias=True))
        model.add(layers.Conv2D(5, (3, 3),  activation='relu', padding='same',use_bias=True))
        model.add(layers.Conv2D(32,(1,1), activation='relu', padding='same', use_bias=True))
        model.add(layers.Conv2DTranspose(3, (9,9), strides=1 ,padding='valid'))

        model.summary()
        self.model = model

class Train():
    def __init__(self):
        self.SRCNN = SRCNN()

    def train(self):
        # load data set
        train_X, train_Y, test_X, test_Y = setdata.setdata()
        # check point
        check_path = './ckpt/cp.ckpt'
        if (retrain == 0):
            self.SRCNN.model.load_weights(check_path)
            print('load checkpoint successfully')
        save_model = keras.callbacks.ModelCheckpoint(check_path,verbose=1,save_weights_only=1, save_freq=10)
        # train
        self.SRCNN.model.compile(optimizer='adam', loss='mean_squared_error')
        self.SRCNN.model.fit(train_X,train_X,epochs=epoch,callbacks=[save_model])
        # train
        img = self.SRCNN.model.predict(train_X)
        img_test =self.SRCNN.model.predict(test_X)
        # cal psnr
        psnr =tf.image.psnr(train_X,img,max_val=1.0)
        psnr_test =tf.image.psnr(test_X,img_test,max_val=1.0)
        print('psnr is %f'%(tf.reduce_mean(psnr)))
        print('psnr in test set is %f' % (tf.reduce_mean(psnr_test)))
app = Train()
app.train()