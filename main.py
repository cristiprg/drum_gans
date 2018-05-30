from data import data_util
import matplotlib.pyplot as plt
import numpy as np

#
# X, Y = data_util.get_spectrograms()
#
# plt.imshow(X[:1000], cmap='hot')
# plt.show()

import data.cnn_data_generator
import data.import_smt


spectrograms, total_len = data.import_smt.load_smt_dataset("./data/smt_spectrograms.h5")
print "len = ", len(spectrograms)


train_ids = np.arange(1000)
#
training_generator = data.cnn_data_generator.DataGenerator(train_ids, spectrograms=spectrograms, dim=(10, 1024, 1), shuffle=False)
