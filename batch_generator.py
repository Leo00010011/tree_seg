import threading
import random
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def form_batch(X, y, batch_size,img_rows,img_cols,num_channels):
    X_batch = np.zeros((batch_size, img_rows, img_cols, num_channels))
    y_batch = np.zeros((batch_size, img_rows, img_cols,1))
    X_height = X.shape[2]
    X_width = X.shape[3]

    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)
        random_image = random.randint(0, X.shape[0] - 1)
        X_batch[i] = np.transpose(X[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols]
                                  ,(1,2,0))
        y_batch[i,:,:,0] = y[random_image, random_height: random_height + img_rows, random_width: random_width + img_cols]

    return X_batch, y_batch


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def batch_generator(X, y, batch_size,img_rows,img_cols,num_channels, horizontal_flip=False, vertical_flip=False, swap_axis=False):
    while True:
        X_batch, y_batch = form_batch(X, y, batch_size ,img_rows,img_cols,num_channels)

        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            yb = y_batch[i]

            if horizontal_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 1)
                    yb = flip_axis(yb, 1)

            if vertical_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 0)
                    yb = flip_axis(yb, 0)

            if swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(0, 1)
                    yb = yb.swapaxes(0, 1)

            X_batch[i] = xb
            y_batch[i] = yb
        yield X_batch, y_batch



import numpy as np
import keras
import itertools

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,X, y, batch_size,img_rows,img_cols,num_channels,steps_per_epoch, horizontal_flip=False, vertical_flip=False, swap_axis=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_channels = num_channels
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.swap_axis = swap_axis
        self.steps_per_epoch = steps_per_epoch
        self.current_batches = []
        self.on_epoch_end()
        self.gen = batch_generator(self.X,self.y,self.batch_size,self.img_rows,self.img_cols,self.num_channels,self.horizontal_flip,self.vertical_flip,self.swap_axis)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        return next(self.gen)


import numpy as np
import keras
import itertools

class DataGeneratorCrop(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,X, y, batch_size,img_rows,img_cols,num_channels,steps_per_epoch,crop_left = 0,crop_right = 0,crop_up = 0,crop_down = 0, horizontal_flip=False, vertical_flip=False, swap_axis=False):
        self.X = X
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_up = crop_up
        self.crop_down = crop_down
        self.y = y
        self.batch_size = batch_size
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_channels = num_channels
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.swap_axis = swap_axis
        self.steps_per_epoch = steps_per_epoch
        self.current_batches = []
        self.on_epoch_end()
        self.gen = batch_generator(self.X,self.y,self.batch_size,self.img_rows,self.img_cols,self.num_channels,self.horizontal_flip,self.vertical_flip,self.swap_axis)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.steps_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_x, batch_y = next(self.gen)
        if not (self.crop_up == 0 and self.crop_down == 0 and self.crop_left == 0 and self.crop_right == 0):
          batch_y = batch_y[:,self.crop_up:-self.crop_down,self.crop_left:-self.crop_right,:]
        return batch_x, batch_y
    
    