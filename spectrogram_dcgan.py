import numpy as np

import data.cnn_data_generator
import data.import_smt
from dcgan import DCGAN
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class Spectrogram_dcgan():
    def __init__(self, j=10, batch_size=128, target="HH"):
        self.training_generator = None
        self.test_generator = None
        self.batch_size = batch_size
        self.target = target

        self.spec_cols = 1024
        self.spec_length = j

        self.DCGAN = DCGAN(img_rows=self.spec_length, img_cols=self.spec_cols, channel=1)
        self.discriminator = self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

        self.data_size = 0
        self.load_data()

    def load_data(self):
        # Prepare data, adapted from spectrogram.cnn.py
        spectrograms, num_frames = data.import_smt.load_smt_dataset("./data/smt_spectrograms.h5")
        data_size = num_frames - (self.spec_cols - 1) * len(
            spectrograms)  # you lose last (j-1) frames at the end of the spectrograms
        train_size = int(data_size * 0.7)

        train_ids = np.arange(start=1, stop=train_size)
        self.training_generator = data.cnn_data_generator.DataGenerator(train_ids, spectrograms=spectrograms, spec_width=self.spec_length,
                                                                        num_channels=1, shuffle=True,
                                                                        n_classes=2, batch_size=self.batch_size,
                                                                        target=self.target)

        test_ids = np.arange(start=train_size, stop=data_size)
        self.test_generator = data.cnn_data_generator.DataGenerator(test_ids, spectrograms=spectrograms, spec_width=self.spec_length,
                                                                    num_channels=1, shuffle=True,
                                                                    n_classes=2, batch_size=self.batch_size,
                                                                    target=self.target)
        self.data_size = data_size

        # train_specs, test_specs = data.import_smt.load_smt_train_test_std("./data/smt_spectrograms.h5", test_size=0)
        # train_num_frames = data.import_smt.count_total_frames(train_specs)
        # test_num_frames = data.import_smt.count_total_frames(test_specs)
        #
        # train_data_size = train_num_frames - (self.spec_length - 1) * len(
        #     train_specs)  # you lose last (j-1) frames at the end of the spectrograms
        # # test_data_size = test_num_frames - (self.spec_length - 1) * len(test_specs)
        #
        # self.training_generator = data.cnn_data_generator.DataGenerator(np.arange(train_data_size), spectrograms=train_specs,
        #                                                            spec_width=self.spec_length, num_channels=1,
        #                                                            shuffle=True, n_classes=2,
        #                                                            batch_size=self.batch_size, target=self.target)
        # # self.test_generator = data.cnn_data_generator.DataGenerator(np.arange(test_data_size), spectrograms=test_specs,
        # #                                                        spec_width=j, num_channels=1, shuffle=True,
        # #                                                        n_classes=2, batch_size=self.batch_size,
        # #                                                        target=self.target)
        #
        # self.train_data_size = train_data_size

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            # images_train = self.x_train[np.random.randint(0,
            #     self.x_train.shape[0], size=batch_size), :, :, :]

            images_train, _ = self.training_generator.__getitem__(i)

            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'spec.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                # filename = "mnist_%d.png" % step
                base_filename = "spectrogram_%d" % step
            images = self.generator.predict(noise)
        else:
            X, _ = self.training_generator.__getitem__(0)
            i = np.random.randint(0, X.shape[0], samples)
            images = X[i, :, :, :]

        # plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            filename = "%s_%d.png" % (base_filename, i)
            plt.imshow(np.flip(
                        images[i]
                       .T
                       .reshape((self.spec_cols, self.spec_length)),
                        0)  # flip the rows, such that in the image the lowest freqs will be down
                       , cmap='gray')

            if save2file:
                plt.savefig(filename)
                plt.close('all')

        #     plt.subplot(4, 4, i + 1)
        #     image = images[i, :, :, :]
        #     image = np.reshape(image, [self.spec_length, self.spec_cols])
        #     plt.imshow(image, cmap='gray')
        #     plt.axis('off')
        # plt.tight_layout()
        # if save2file:
        #     plt.savefig(filename)
        #     plt.close('all')
        # else:
        #     plt.show()

if __name__ == '__main__':
    spectrogram_dcgan = Spectrogram_dcgan(j=16, batch_size=128, target="HH")
    spectrogram_dcgan.train(train_steps=900, batch_size=128, save_interval=2)


