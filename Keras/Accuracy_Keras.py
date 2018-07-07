import matplotlib.pyplot as plt
from keras.datasets import cifar10
import numpy as np
import datetime
import keras

from keras.utils import np_utils
from keras.models import  load_model

model = load_model('Ass3.hdf5')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

mean_train = np.mean(x_train,axis=0)
mean_test = np.mean(x_test,axis=0)

x_train -= mean_train
x_test -= mean_test

x_train  /= np.std(x_train,axis=0)
x_test /= np.std(x_test,axis=0)    

score = model.evaluate(x_test, np_utils.to_categorical(y_test, 10), verbose=1)
print('ACCR:', score[1])

classes = model.predict_classes(x_test, verbose=1)
accuracy_per_class = [0.] * 10

for i in range(classes.shape[0]):
    if classes[i] == y_test[i]:
        accuracy_per_class[int(y_test[i])] += 1
for i in range(10):
    accuracy_per_class[i] /= 1000.0

c = ['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck']
for i in range(10):
    print("\nCCRn of %s is %f" % (c[i], accuracy_per_class[i]))

x = input()