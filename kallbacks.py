import keras
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

import logging

class AccuracyHistory(Callback):

    def __init__(self):
        super(AccuracyHistory, self).__init__()
        self.current_batch = 0

    def on_train_begin(self, logs={}):
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        if self.current_batch == 0 or (self.current_batch+1) % 50:
            self.acc.append(logs.get('acc'))

        self.current_batch += 1


class BatchTensorBoard(keras.callbacks.TensorBoard):
    def on_batch_end(self, batch, logs=None):
        if batch == 0 or (batch+1) % 10 == 0:
            super(BatchTensorBoard, self).on_epoch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        pass


class AUCHistory(Callback):
    """
    Computes area under ROC curve.
    Copied from here:  https://gist.github.com/smly/d29d079100f8d81b905e
    """
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.aucs = []

    def on_batch_end(self, batch, logs=None):
        if batch == 0 or (batch +1) % self.interval == 0:
            y_pred = self.model.predict_proba(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            self.aucs.append(score)
            logging.info("interval evaluation - batch: {:d} - score: {:.6f}".format(batch, score))