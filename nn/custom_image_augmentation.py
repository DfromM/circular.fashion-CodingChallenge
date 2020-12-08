import keras.utils
import numpy as np
from keras.preprocessing.image import random_rotation, random_shift, \
    random_zoom


class ClothImageDataGenerator(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size=1000, shuffle=True,
                 alter_background=True, max_rotation=0, max_shift=0.0,
                 max_zoom=0.0, random_noise=False):

        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.alter_background = alter_background
        self.max_rotation = max_rotation
        self.max_shift = max_shift
        self.max_zoom = max_zoom
        self.with_random_noise = random_noise

        self.transform()

    def transform(self):
        if self.alter_background:
            self.alter_image_backgrounds()
            print("backgrounds finished")

        if self.max_rotation > 0:
            self.make_rotations()
            print("rotation finished")

        if self.max_shift > 0:
            self.shift_images()
            print("shift finished")

        if self.with_random_noise:
            self.add_random_noise()
            print("noise finished")

        if self.max_zoom > 0:
            self.zoom()
            print("zoom finished")

        if self.shuffle:
            self.shuffle_all_signals()
            print("shuffle finished")

    def alter_image_backgrounds(self):
        for i in range(self.x.shape[0]):
            if i % 3 == 0:
                rand = np.random.uniform(0, 1)
                for j in range(self.x.shape[1]):
                    for k in range(self.x.shape[2]):
                        if self.x[i, j, k] == 1:
                            self.x[i, j, k] = rand

    def make_rotations(self):
        for i in range(self.x.shape[0]):
            random_rotation(self.x[i], self.max_rotation, row_axis=0,
                            col_axis=1, channel_axis=2)

    def shift_images(self):
        for i in range(self.x.shape[0]):
            random_shift(self.x[i], self.max_shift, self.max_shift, row_axis=0,
                         col_axis=1, channel_axis=2)

    def shuffle_all_signals(self):
        permutation = np.random.permutation(len(self.x))
        shuffled_x = np.empty(self.x.shape, dtype=self.x.dtype)
        shuffled_y = np.empty(self.y.shape, dtype=self.y.dtype)
        for old_index, new_index in enumerate(permutation):
            shuffled_x[new_index] = self.x[old_index]
            shuffled_y[new_index] = self.y[old_index]

    def zoom(self):
        for i in range(self.x.shape[0]):
            random_zoom(self.x[i], (self.max_zoom, self.max_zoom), row_axis=0, col_axis=1,
                        channel_axis=2)

    def add_random_noise(self):
        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                for k in range(self.x.shape[2]):
                    self.x[i, j, k] = self.x[i, j, k] + \
                                      np.random.uniform(-0.05, 0.05)

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_all_signals()

    def __len__(self):
        # number of batches per epoch
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, batch_number):
        # one batch of data
        batch_x = self.x[batch_number * self.batch_size: (
                                                                 batch_number + 1) * self.batch_size]
        batch_y = self.y[batch_number * self.batch_size: (
                                                                 batch_number + 1) * self.batch_size]

        return batch_x, batch_y