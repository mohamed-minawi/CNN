
import matplotlib.pyplot as plt
import numpy as np
import datetime
import keras

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.initializers import he_normal
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:

def pre_process_data(x_train, x_test, y_train, y_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    mean_train = np.mean(x_train,axis=0)
    mean_test = np.mean(x_test,axis=0)

    x_train -= mean_train
    x_test -= mean_test

    x_train  /= np.std(x_train,axis=0)
    x_test /= np.std(x_test,axis=0)
    
    return x_train, x_test, y_train, y_test


# In[ ]:


def create_model(x_train):
    
    model = Sequential()
    
    model.add(Conv2D(192, (5, 5), padding='same', input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(160, (1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(96, (1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))
    
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))
    
    model.add(Dropout(0.5))
    
    model.add(Conv2D(192, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Flatten())
       
    model.add(Dense(500))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))
        
    nadam_opt = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, schedule_decay=0.004)

    model.compile(loss='categorical_crossentropy', optimizer = nadam_opt, metrics=['accuracy'])
    
    return model




def predict_classes(model, x_test, y_test):
    classes = model.predict_classes(x_test, verbose=1)
    y_test =  np.array([np.argmax(y, axis=None, out=None) for y in y_test])
    accuracy_per_class = [0.] * 10
    
    
    for i in range(classes.shape[0]):
        if classes[i] == y_test[i]:
            accuracy_per_class[int(y_test[i])] += 1
    for i in range(10):
        accuracy_per_class[i] /= 1000.0

    c = ['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    for i in range(10):
        print("\nCCRn of %s is %f" % (c[i], accuracy_per_class[i]))


# In[ ]:


def fit_model(x_train, y_train, model, batch_size, epochs):
    datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False, 
        featurewise_std_normalization=False, 
        samplewise_std_normalization=False,  
        zca_whitening=False, 
        rotation_range=5,  
        width_shift_range=0.16,  
        height_shift_range=0.16, 
        horizontal_flip=True, 
        vertical_flip=False) 
    
    datagen.fit(x_train[:40000])
    checkpointer = ModelCheckpoint(filepath="./Ass3.hdf5", verbose=1, save_best_only=True, monitor='val_acc')

    history = model.fit_generator(datagen.flow(x_train[:40000], y_train[:40000], batch_size=batch_size), epochs=epochs, steps_per_epoch = 391,
                        validation_data=(x_train[40000:], y_train[40000:]),workers=8, callbacks=[checkpointer])
    
    return history, model



batch_size = 128
num_classes = 10
epochs = 20

np.random.seed(7)


# In[ ]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test, y_train, y_test = pre_process_data(x_train, x_test, y_train, y_test)
model = create_model(x_train)
history, model = fit_model(x_train, y_train, model,batch_size, epochs)



model = load_model("Ass3.hdf5")
history, model = fit_model(x_train, y_train, model,batch_size, epochs)
model.save('currentmodel.h5')

accuracy = history.history['acc']
val_accuracy = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

