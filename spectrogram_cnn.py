import keras
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, InputLayer
from keras.models import Sequential
from sklearn.model_selection import train_test_split

import matplotlib.pylab as plt
import numpy as np
import datetime

import data.cnn_data_generator
import data.import_smt
import kallbacks


def get_cnn_model():
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(10, kernel_size=(5, 32), strides=(5, v),
                     activation='relu',
                     padding='same'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(20, (5, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, e)))

    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))

    return model

def get_nn_model():
    model = Sequential()
    model.add(InputLayer(input_shape=(j*1024,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    return model


# Prepare variables
DEBUG = False # For checking Tensorboard. This will not use a generator but will load the entire dataset in memory.
nr_workers = 12 # IMPORTANT!!!!!!!!!

j = 16  # @param
v = 5  # @param
e = 2  # @param
num_classes = 2
batch_size = 128
epochs = 10
class_imbalance = 0.14
target = "KD"
input_shape = (j, 1024, 1)
# input_shape = (j*1024,)
nn_type = "NN"
num_channels = 1
seed = 42

# Declare the CNN model

if nn_type is "CNN":
    model = get_cnn_model()
    num_channels = 1
elif nn_type is "NN":
    model = get_nn_model()
    num_channels = 0
else:
    model = None


# Prepare data
# spectrograms, num_frames = data.import_smt.load_smt_dataset("./data/smt_spectrograms.h5")
train_specs, test_specs = data.import_smt.load_smt_train_test_std("./data/smt_spectrograms.h5", seed=seed)
train_num_frames = data.import_smt.count_total_frames(train_specs)
test_num_frames = data.import_smt.count_total_frames(test_specs)

train_data_size = train_num_frames - (j-1) * len(train_specs)  # you lose last (j-1) frames at the end of the spectrograms
test_data_size = test_num_frames - (j-1) * len(test_specs)

training_generator = data.cnn_data_generator.DataGenerator(np.arange(train_data_size), spectrograms=train_specs, spec_width=j, num_channels=num_channels, shuffle=True, n_classes=num_classes, batch_size=batch_size, target=target)
test_generator = data.cnn_data_generator.DataGenerator(np.arange(test_data_size), spectrograms=test_specs, spec_width=j, num_channels=num_channels, shuffle=True, n_classes=num_classes, batch_size=batch_size, target=target)


model.add(Dense(1, activation='sigmoid'))
loss = keras.losses.binary_crossentropy
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

tensorboard_id = datetime.datetime.now().isoformat() + "_" + target + "_" + nn_type
tensorboard = kallbacks.BatchTensorBoard(log_dir='./logs/' + tensorboard_id,
    histogram_freq=1, batch_size=batch_size, write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
history = kallbacks.AccuracyHistory()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=2, verbose=1, mode='min')


f, (ax1, ax2) = plt.subplots(2)
if DEBUG:
    # Note: for visualing histograms, it is required to NOT have a generator
    nr_debug_batches = 10
    X_train, y_train, X_test, y_test = [], [], [], []

    # TODO: find a way to make this for loop faster
    for i in range(nr_debug_batches):
        X, y = test_generator.__getitem__(i)
        X_test.extend(X)
        y_test.extend(y)

        X, y = training_generator.__getitem__(i)
        X_train.extend(X)
        y_train.extend(y)

    # Convert to numpy arrays
    X_train, y_train,X_test ,y_test  = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

    if nn_type is "NN":  # TODO, de ce naiba nu se schimba shapeu?
        X_train = X_train.reshape((-1, j * 1024))
        X_test = X_test.reshape((-1, j*1024))

    auc_history = kallbacks.AUCHistory((X_test, y_test))

    model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=1,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[history, tensorboard, auc_history],
          shuffle=True,
          class_weight={0: class_imbalance, 1: (1-class_imbalance)}
              )

    ax2.plot(range(0, len(auc_history.aucs)), auc_history.aucs)
    ax2.set_ylabel('AUC score')

else:

    model.fit_generator(
        generator=training_generator,
        validation_data=test_generator,
        epochs=epochs,
        verbose=2,
        callbacks=[history, early_stopping],
        use_multiprocessing=True,
        workers=nr_workers,
        # validation_split=0.3,
        class_weight={0: class_imbalance, 1: (1-class_imbalance)}
    )

score = model.evaluate_generator(generator=test_generator, use_multiprocessing=True, workers=nr_workers)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
ax1.plot(range(0, len(history.acc)), history.acc)
ax1.set_xlabel('Step')
ax1.set_ylabel('Accuracy')

plt.show()
