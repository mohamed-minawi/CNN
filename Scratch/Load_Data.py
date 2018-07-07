import os
import pickle
import numpy as np
from scipy.ndimage import rotate 
import scipy.ndimage as ndi

def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
    return X, Y

def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)    
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
  
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def augment_image(x,rotation_range=0,height_shift_range=0,
                  width_shift_range=0,img_row_axis=1,img_col_axis=2,
                  img_channel_axis=0,horizontal_flip=False,vertical_flip=False):

    if rotation_range != 0:
        theta = np.deg2rad(np.random.uniform(rotation_range, rotation_range))
    else:
        theta = 0

    if height_shift_range != 0:
        tx = np.random.uniform(-height_shift_range, height_shift_range)
        if height_shift_range < 1:
            tx *= x.shape[img_row_axis]
    else:
        tx = 0

    if width_shift_range != 0:
        ty = np.random.uniform(-width_shift_range, width_shift_range)
        if width_shift_range < 1:
            ty *= x.shape[img_col_axis]
    else:
        ty = 0

    transform_matrix = None
    if theta != 0:
        transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                            [np.sin(theta), np.cos(theta), 0],
                                            [0, 0, 1]])

    if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)


    if horizontal_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_col_axis)

    if vertical_flip:
        if np.random.random() < 0.5:
            x = flip_axis(x, img_row_axis)

    if transform_matrix is not None:
        h, w = x.shape[img_row_axis], x.shape[img_col_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_axis)

    return x



def augment_batch(images,rotation_range=0,height_shift_range=0,
                  width_shift_range=0,img_row_axis=1,img_col_axis=2,
                  img_channel_axis=0,horizontal_flip=False,vertical_flip=False):
    x = np.array(images)
    indx = images.shape[0]
    for i in range(0,indx):
        #if i%5000 == 0:
         #   print ('Augmented %d images' %(i))

        x[i] = augment_image(images[i],
         rotation_range=rotation_range,
         img_row_axis=img_row_axis,
         img_col_axis=img_col_axis,
         img_channel_axis=img_channel_axis,
         horizontal_flip = horizontal_flip,
         vertical_flip = vertical_flip,
         height_shift_range = height_shift_range,
         width_shift_range = width_shift_range)

    return x

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000, augment = False):
    
    cifar10_dir = './cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    X_train  /= np.std(X_train,axis=0)
    X_test /= np.std(X_test,axis=0)
    X_val /= np.std(X_val,axis=0)

    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }