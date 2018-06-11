from _curses import filter

import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/")
# sample_image = mnist.train.next_batch(1)[0]
# print sample_image.shape
# plt.imshow(sample_image.reshape((28, 28)))

import data.cnn_data_generator
import data.import_smt

j = 16  # @param
v = 5  # @param
e = 2  # @param
num_classes = 2
batch_size = 64
epochs = 10
class_imbalance = 0.14
target = "KD"
input_shape = (j, 1024, 1)
# input_shape = (j*1024,)
nn_type = "CNN_Conditional_Discriminator"
num_channels = 1
seed = 42

# Prepare data
# spectrograms, num_frames = data.import_smt.load_smt_dataset("./data/smt_spectrograms.h5")
train_specs, test_specs = data.import_smt.load_smt_train_test_std("./data/smt_spectrograms.h5", seed=seed)

gan_train_ratio = 0.00
nr_gan_train_specs = int(len(train_specs) * gan_train_ratio)
gan_train_specs, cnn_train_specs = train_specs[:nr_gan_train_specs], train_specs[nr_gan_train_specs:]

# From here, train means CNN train, not GAN train
train_num_frames = data.import_smt.count_total_frames(cnn_train_specs)
test_num_frames = data.import_smt.count_total_frames(test_specs)

train_data_size = train_num_frames - (j-1) * len(cnn_train_specs)  # you lose last (j-1) frames at the end of the spectrograms
test_data_size = test_num_frames - (j-1) * len(test_specs)

training_generator = data.cnn_data_generator.DataGenerator(np.arange(train_data_size), spectrograms=cnn_train_specs, spec_width=j, num_channels=num_channels, shuffle=True, n_classes=num_classes, batch_size=batch_size, target=target)
test_generator = data.cnn_data_generator.DataGenerator(np.arange(test_data_size), spectrograms=test_specs, spec_width=j, num_channels=num_channels, shuffle=True, n_classes=num_classes, batch_size=batch_size, target=target)

print("len(training_generator)=", len(training_generator))

def discriminator(images, reuse=False):

    # if (reuse):
    #     tf.get_variable_scope().reuse_variables()
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

        # First conv layer and pool layers
        # Size : j*1024*1
        d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32],
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32],
                               initializer=tf.constant_initializer(0))

        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Second conv layer, almost same
        # Size: 8*512*32
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64],
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64],
                               initializer=tf.constant_initializer(0))

        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Fully connected layer 1
        # Size: 4*256*64
        d_w3 = tf.get_variable('d_w3', [4 * 256 * 64, 1024],
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024],
                               initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, 4 * 256 * 64])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        # Fully connected layer 2
        d_w4 = tf.get_variable('d_w4', [1024, 1],
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1],
                               initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4)
        d4 = d4 + d_b4

        return d4


def generator(z, batch_size, z_dim):
    g_w1 = tf.get_variable('g_w1', [z_dim, j*1024*4], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('b_w1', [j*1024*4], initializer=tf.truncated_normal_initializer(stddev=0.02))

    g1 = tf.matmul(z, g_w1)
    g1 = g1 + g_b1
    g1 = tf.reshape(g1, [-1, j*2, 1024*2, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [j*2, 1024*2])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim / 2, z_dim / 4], dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim / 4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [j*2, 1024*2])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    return g4


# batch_size = 64

# placeholder for noise generator
z_dimensions = 100
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name="z_placeholder")

x_placeholder = tf.placeholder(tf.float32, shape=[None, j, 1024, 1], name="x_placeholder")

Gz = generator(z_placeholder, batch_size, z_dimensions)  # Generated images
Dx = discriminator(x_placeholder)
Dg = discriminator(Gz, reuse=True)

# Define the losses - take care here!
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=Dx, labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=Dg, labels=tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=Dg, labels=tf.ones_like(Dg)))

# Get handles for the weights in discriminator and generator
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

# Define the training for discriminator and generator
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
tf.summary.image('Generated_images', images_for_tensorboard, 5)

merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logdir, sess.graph)

    # Trick: pretrain the discriminator only
    steps_pre_train_d = 100
    for i in range(steps_pre_train_d):
        # real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
        real_image_batch, _ = training_generator.__getitem__(i)
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

        _, _, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                              feed_dict={x_placeholder: real_image_batch,
                                                         z_placeholder: z_batch})

        if i % 50 == 0:
            print("dLossReal:", dLossReal, " dLossFake:", dLossFake)

    # Main loop - train generator and discriminator
    for i in range(steps_pre_train_d, len(training_generator)):

        # Train discriminator on real and fake data
        # real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
        real_image_batch, _ = training_generator.__getitem__(i)
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

        _, _, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                              feed_dict={x_placeholder: real_image_batch,
                                                         z_placeholder: z_batch})

        # Train generator
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

        if i % 10 == 0:
            # Update Tensorboard
            z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
            summary = sess.run(merged, feed_dict={x_placeholder: real_image_batch,
                                                  z_placeholder: z_batch})
            writer.add_summary(summary, i)
            writer.flush()

        if i % 100 == 0:
            # Generate some images for the fun of it
            print("Iteration: ", i)
            z_batch = np.random.normal(0, 1, size=[1, z_dimensions])
            img = sess.run(Gz, feed_dict={z_placeholder: z_batch})

            plt.imshow(img.T.reshape([1024, j]), cmap='gray')
            plt.savefig("image_%d.png" % i)

            result = discriminator(x_placeholder)
            estimate = sess.run(result, feed_dict={x_placeholder: img.reshape([1, j, 1024, 1])})
            print("Estimate: ", estimate)
