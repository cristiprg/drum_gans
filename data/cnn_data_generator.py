import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    Example adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    """
    def __init__(self, list_IDs, spectrograms=None, target="HH", batch_size=32, dim=(32,32,32),
                 n_classes=2, shuffle=True):
        'Initialization'

        if spectrograms is None:
            raise ValueError("Cannot generate data: spectrograms is None")
        if len(list_IDs) <= batch_size:
            raise ValueError("Cannot generate data: len(list_IDs) = " + str(len(spectrograms)))

        #TODO: more sanity checks on spectrograms?
        self.spectrograms = spectrograms
        self.target = target
        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        self.list_IDs = list_IDs
#         self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

        self.__build_internal_index()

    def __build_internal_index(self):
        """
        Builds a map from ID -> (spectrogram_index, row_number). This is necessary because each spectrogram does not
        have a fixed length length
        """
        i = 0
        curr_spec_index = 0
        self.LUT = []
        while curr_spec_index < len(self.spectrograms):
            curr_spec_len = self.spectrograms[curr_spec_index].shape[0]
            if i < curr_spec_len - self.dim[0] + 1:  # subtract the width of the image
                # self.LUT[i] = (curr_spec_index, i)
                self.LUT.append((curr_spec_index, i))
                i += 1
            else:
                i = 0
                curr_spec_index += 1

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization, Python 2
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2]))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            spectrogram_index, row_number = self.LUT[ID]
            # X[i] = self.spectrograms[ID:(ID+self.dim[0])].reshape(self.dim)
            X[i] = self.spectrograms[spectrogram_index].iloc[row_number:row_number+self.dim[0], 0:self.dim[1]]\
                .values.reshape(self.dim)

            # Store class
            y[i] = self.spectrograms[spectrogram_index].iloc[row_number][self.target]

        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y


# Small test, consecutive have to also look consecutive, check with shuffle = False
# X, y = training_generator.__getitem__(0)
# for i in range(10, 20):
#     plt.imshow(X[i][0:10, 0:200].reshape((10, 200)).T, vmin=0, vmax=1.2)
#     plt.show()
