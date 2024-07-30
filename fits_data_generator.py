import numpy as np
import os
from tensorflow.keras.utils import Sequence
import astropy.io.fits as fits

class FITSDataGenerator(Sequence):
    """
    A data generator class inheriting from Keras' Sequence, to be used for generating batches of
    FITS file data for training a model.

    Attributes:
        noisy_dir (str): Directory containing simulated galaxy images injected into sky background.
        clean_dir (str): Directory containing simulated galaxy images.
        indices (np.array): Array of indices to use for generating data batches.
        batch_size (int): Size of data batches.
        rescale (float): Rescaling factor to apply to data.
    """
    def __init__(self, noisy_dir, clean_dir, indices, batch_size=32, shuffle=True, rescale=1.):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rescale = rescale
        self.noisy_paths = np.array([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith('.fits')])
        self.clean_paths = np.array([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.fits')])
        self.noisy_paths.sort()
        self.clean_paths.sort()

    def __len__(self):
        # Calculate the number of batches per epoch.
        return int(np.floor(len(self.indices) / self.batch_size))


    def __getitem__(self, index):
        # Generate one batch of data.
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        noisy_batch_paths = self.noisy_paths[batch_indices]
        clean_batch_paths = self.clean_paths[batch_indices]
        noisy = self.__data_generation(noisy_batch_paths)
        clean = self.__data_generation(clean_batch_paths)
        return noisy, clean

    def __data_generation(self, batch_paths):
        # Generate data for the batch based on file paths.
        X = np.array([self.preprocess_fits(fits.open(file_path)[0].data) for file_path in batch_paths])
        return X

    def preprocess_fits(self, data):
        # Preprocess the data by rescaling and reshaping.
        data = self.rescale * data.astype('float32')
        return data.reshape(data.shape + (1,))  # Assuming data is 2D, add channel dimension.
