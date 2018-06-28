import multiprocessing
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

DRY_RUN = False
DBFILE = "smt_train_logging_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
z_dimensions = 100

# Prepare data
def prepare_data(seed=42):
    """
    Returns 3 generators, one for GAN training, one for CNN trainig and one for testing.
    The split is done in the following manner:
    100% - 25% CNN testing
         - 75% training - 50% GAN training
                        - 50% CNN training
    The ratios can be easily changed.
    """

    # spectrograms, num_frames = data.import_smt.load_smt_dataset("./data/smt_spectrograms.h5")
    train_specs, test_specs = data.import_smt.load_smt_train_test_std("./data/smt_spectrograms.h5", seed=seed)

    if DRY_RUN:
        train_specs = train_specs[:256]
        test_specs = test_specs[:256]

    # TODO: change this to random split, not just consecutive split
    gan_train_ratio = 0.5
    nr_gan_train_specs = int(len(train_specs) * gan_train_ratio)
    gan_train_specs, cnn_train_specs = train_specs[:nr_gan_train_specs], train_specs[nr_gan_train_specs:]

    gan_train_num_frames = data.import_smt.count_total_frames(gan_train_specs)
    train_num_frames = data.import_smt.count_total_frames(cnn_train_specs)
    test_num_frames = data.import_smt.count_total_frames(test_specs)

    gan_train_size = gan_train_num_frames - (j - 1) * len(gan_train_specs)
    train_data_size = train_num_frames - (j - 1) * len(
        cnn_train_specs)  # you lose last (j-1) frames at the end of the spectrograms
    test_data_size = test_num_frames - (j - 1) * len(test_specs)

    gan_training_generator = data.cnn_data_generator.DataGenerator(np.arange(gan_train_size),
                                                                   spectrograms=gan_train_specs, spec_width=j,
                                                                   num_channels=num_channels, shuffle=True,
                                                                   n_classes=num_classes, batch_size=batch_size,
                                                                   target=target)
    training_generator = data.cnn_data_generator.DataGenerator(np.arange(train_data_size), spectrograms=cnn_train_specs,
                                                               spec_width=j, num_channels=num_channels, shuffle=True,
                                                               n_classes=num_classes, batch_size=batch_size,
                                                               target=target)
    test_generator = data.cnn_data_generator.DataGenerator(np.arange(test_data_size), spectrograms=test_specs,
                                                           spec_width=j, num_channels=num_channels, shuffle=True,
                                                           n_classes=num_classes, batch_size=batch_size, target=target)

    return gan_training_generator, training_generator, test_generator


def resettable_metric(metric, scope_name, **metric_args):
    '''
    Originally from https://github.com/tensorflow/tensorflow/issues/4814#issuecomment-314801758
    '''
    with tf.variable_scope(scope_name) as scope:
        metric_op, update_op = metric(**metric_args)
        v = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(v)
    return metric_op, update_op, reset_op


def discriminator(images, reuse=False):
    # if (reuse):
    #     tf.get_variable_scope().reuse_variables()
    # with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
    with tf.variable_scope("discriminator_scope", reuse=reuse):
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
        d_w4 = tf.get_variable('d_decision_w4', [1024, 1],
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_decision_b4', [1],
                               initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4)
        d4 = d4 + d_b4

        return d4


def generator(z, batch_size, z_dim, reuse=False):
    with tf.variable_scope("generator_scope", reuse=reuse):
        g_w1 = tf.get_variable('g_w1', [z_dim, j*1024*4], dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.02))
        g_b1 = tf.get_variable('g_b1', [j*1024*4], initializer=tf.truncated_normal_initializer(stddev=0.02))

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


def define_cnn_graph_structure():
    x_placeholder = tf.placeholder(tf.float32, shape=[None, j, 1024, 1], name="x_placeholder")
    y_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name="y_placeholder")

    Dx = discriminator(x_placeholder, reuse=False)

    tvars = tf.trainable_variables()
    d_vars = [var for var in tvars if 'd_' in var.name]
    d_decision_vars = [var for var in tvars if 'd_decision' in var.name]

    d_cnn_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=Dx, targets=y_placeholder,
                                                                         pos_weight=1/0.14))

    d_cnn_trainer = tf.train.AdamOptimizer(0.0001).minimize(d_cnn_loss, var_list=d_vars)
    d_decision_trainer = tf.train.AdamOptimizer(0.0001).minimize(d_cnn_loss, var_list=d_decision_vars)

    tf.summary.scalar('CNN_loss', d_cnn_loss)

    return x_placeholder, y_placeholder, Dx, d_cnn_loss, d_cnn_trainer, d_decision_trainer


def define_gan_graph_structure():
    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name="z_placeholder")

    x_placeholder = tf.placeholder(tf.float32, shape=[None, j, 1024, 1], name="x_placeholder")
    y_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name="y_placeholder")

    Gz = generator(z_placeholder, batch_size, z_dimensions, reuse=False)  # Generated images
    Dx = discriminator(x_placeholder, reuse=False)
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

    tf.summary.scalar('Generator_loss', g_loss)
    tf.summary.scalar('Discriminator_loss_real', d_loss_real)
    tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

    images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions, reuse=True)
    tf.summary.image('Generated_images', images_for_tensorboard, 5)

    return x_placeholder, y_placeholder, z_placeholder, Dx, d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake, g_trainer


def perform_gan_training(gan_training_generator):
    print("len(gan_training_generator) == ", len(gan_training_generator))

    x_placeholder, y_placeholder, z_placeholder, Dx, d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake, \
        g_trainer = define_gan_graph_structure()

    checkpoint_name = "./smt_checkpoints/trainable_variables" + "_" + str(train_data_percentage) + "_" + str(seed) + "_GAN" + ".ckpt"
    tensorboard_session_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + str(train_data_percentage) + "_" + str(seed) + "_GAN"
    logdir = "tensorboard/" + str(tensorboard_session_name) + "/"
    # Save variables for the discriminator and the generator
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator_scope')
                                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator_scope'))

    if tf.train.checkpoint_exists(checkpoint_name):
        print "WARNING not performing training! Following checkpoint found, this was already computed, are you sure you still want to recompute? ", checkpoint_name
        return None

    with tf.Session(config=tf.ConfigProto()) as sess:

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(logdir, sess.graph)

        global_step = 0
        print datetime.datetime.utcnow(), "STEP 1: TRAIN GAN (3 epochs)"
        for e in range(3):
            stop = int(len(gan_training_generator) * train_data_percentage / 100.0) if not DRY_RUN else 1
            for i in range(stop):
                # Train discriminator on real and fake data
                # real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
                real_image_batch, _ = gan_training_generator.__getitem__(i)
                z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

                _, _, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                                      feed_dict={x_placeholder: real_image_batch,
                                                                 z_placeholder: z_batch})

                # Train generator
                z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
                sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

                if i % 50 == 0:
                    # Update Tensorboard
                    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
                    summary = sess.run(tf.summary.merge_all(), feed_dict={x_placeholder: real_image_batch,
                                                          z_placeholder: z_batch})
                    writer.add_summary(summary, global_step)
                    writer.flush()

                global_step += 1


        saver.save(sess, checkpoint_name)


def perform_cnn_training(training_generator, test_generator):

    x_placeholder, y_placeholder, Dx, d_cnn_loss, d_cnn_trainer, d_decision_trainer = define_cnn_graph_structure()

    m_op, up_op, reset_op = resettable_metric(tf.metrics.mean, 'foo',
                                              values=tf.nn.sigmoid_cross_entropy_with_logits(
                                                  logits=Dx, labels=y_placeholder))

    roc_m_op, roc_up_op, roc_reset_op = resettable_metric(tf.metrics.auc, 'foo2',
                                                          predictions=tf.nn.sigmoid(Dx), labels=y_placeholder)

    import Queue
    def compute_delta_losses(Q):
        vals = np.array(Q.queue)
        s = 0
        for i in range(1, len(vals)):
            s += vals[i - 1] - vals[i]
        return 1.0 * s / len(vals)

    def compute_roc_auc():
        sess.run(roc_reset_op)
        stop = len(test_generator) if not DRY_RUN else 1
        for i in range(stop):
            real_image_test_batch, y_test_batch = test_generator.__getitem__(j)
            _, _ = sess.run([roc_m_op, roc_up_op], feed_dict={x_placeholder: real_image_test_batch,
                                                              y_placeholder: y_test_batch.reshape((-1, 1))})
        roc = sess.run(roc_m_op)
        return roc

    print("len(training_generator) == ", len(training_generator))
    print("len(test_generator) == ", len(test_generator))

    df_data = np.ndarray(shape=(0, 6))
    # Early stopping delta
    delta_loss = 0.005
    Q = Queue.Queue(maxsize=4)

    saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator_scope'))
    checkpoint_name = "./smt_checkpoints/trainable_variables" + "_" + str(train_data_percentage) + "_" + str(seed) + "_CNN" + ".ckpt"
    gan_checkpoint_name = "./smt_checkpoints/trainable_variables" + "_" + str(train_data_percentage) + "_" + str(seed) + "_GAN" + ".ckpt"

    tensorboard_session_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + str(
        train_data_percentage) + "_" + str(seed) + "_CNN"
    logdir = "tensorboard/" + str(tensorboard_session_name) + "/"
    loss_summary = tf.summary.scalar("cnn_loss", m_op)

    if tf.train.checkpoint_exists(checkpoint_name):
        print "WARNING not performing training! Following checkpoint found, this was already computed, are you sure you still want to recompute? ", checkpoint_name
        return None

    # Incremented every EPOCH
    global_step = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, gan_checkpoint_name)  # Here we have the values for the discriminator from the previous session
        writer = tf.summary.FileWriter(logdir, sess.graph)

        print datetime.datetime.utcnow(), "STEP 2: TRAIN DECISION LAYER ONLY"
        Q.queue.clear()

        for e in range(epochs):
            stop = int(len(training_generator) * train_data_percentage / 100.0) if not DRY_RUN else 1
            for i in range(stop):
                real_image_train_batch, y_train = training_generator.__getitem__(i)
                y_train = y_train.reshape((-1, 1))
                _, dLossReal = sess.run([d_decision_trainer, d_cnn_loss],
                                        feed_dict={x_placeholder: real_image_train_batch,
                                                   y_placeholder: y_train})

            # Test on entire test set
            sess.run(reset_op)
            stop = len(test_generator) if not DRY_RUN else 1
            for i in range(stop):
                real_image_test_batch, y_test_batch = test_generator.__getitem__(i)
                _, _ = sess.run([m_op, up_op], feed_dict={x_placeholder: real_image_test_batch,
                                                          y_placeholder: y_test_batch.reshape((-1, 1))})
            loss = sess.run(m_op)
            print datetime.datetime.utcnow(), "Train percentage: ", train_data_percentage, " Fold: ", seed, " Test epoch:", e, " loss: ", loss
            df_row = np.array([train_data_percentage, seed, 0, e, loss, np.nan]).reshape(-1, 6)
            df_data = np.append(df_data, df_row, axis=0)

            # Update tensorboard
            summary = sess.run(loss_summary)
            writer.add_summary(summary, global_step)
            writer.flush()

            Q.put(loss)

            #  Check the average of last 3 epochs
            if Q.full():
                mean_delta_losses = compute_delta_losses(Q)
                if mean_delta_losses < delta_loss:
                    print "Early stopping: mean_delta_losses = ", mean_delta_losses
                    break

                # Remove one element from the queue
                Q.get()

            global_step += 1

        print datetime.datetime.utcnow(), "STEP 2.1: COMPUTE ROC AUC"
        roc = compute_roc_auc()
        df_row = np.array([train_data_percentage, seed, 1, np.nan, np.nan, roc]).reshape(-1, 6)
        df_data = np.append(df_data, df_row, axis=0)

        print datetime.datetime.utcnow(), "STEP 3: FURTHER TRAIN DISCRIMINATOR"

        sess.run(reset_op)
        Q.queue.clear()
        for e in range(epochs):
            stop = int(len(training_generator) * train_data_percentage / 100.0) if not DRY_RUN else 1
            for i in range(stop):
                real_image_train_batch, y_train = training_generator.__getitem__(i)
                y_train = y_train.reshape((-1, 1))

                _, dLossReal = sess.run([d_cnn_trainer, d_cnn_loss],
                                        feed_dict={x_placeholder: real_image_train_batch,
                                                   y_placeholder: y_train})


            # Test on entire test set
            sess.run(reset_op)
            stop = len(test_generator) if not DRY_RUN else 1
            for i in range(stop):
                real_image_test_batch, y_test_batch = test_generator.__getitem__(i)
                _, _ = sess.run([m_op, up_op], feed_dict={x_placeholder: real_image_test_batch,
                                                          y_placeholder: y_test_batch.reshape((-1, 1))})
            loss = sess.run(m_op)
            print datetime.datetime.utcnow(), "Train percentage: ", train_data_percentage, " Fold: ", seed, " Test epoch:", e, " loss: ", loss
            df_row = np.array([train_data_percentage, seed, 1, e, loss, np.nan]).reshape(-1, 6)
            df_data = np.append(df_data, df_row, axis=0)

            # Update tensorboard
            summary = sess.run(loss_summary)
            writer.add_summary(summary, global_step)
            writer.flush()

            Q.put(loss)
            #  Check the average of last 3 epochs
            if Q.full():
                mean_delta_losses = compute_delta_losses(Q)
                if mean_delta_losses < delta_loss:
                    print "Early stopping: mean_delta_losses = ", mean_delta_losses
                    break

                # Remove one element from the queue
                Q.get()

            global_step += 1

        print datetime.datetime.utcnow(), "STEP 3.1: COMPUTE ROC AUC"
        roc = compute_roc_auc()
        df_row = np.array([train_data_percentage, seed, 1, np.nan, np.nan, roc]).reshape(-1, 6)
        df_data = np.append(df_data, df_row, axis=0)

        saver.save(sess, checkpoint_name)

        return df_data


def run_training_step(q):
        print datetime.datetime.utcnow(), "STEP 0: PREPARE DATA FOLD ", seed
        gan_training_generator, training_generator, test_generator = prepare_data(
            seed=seed)

        perform_gan_training(gan_training_generator)

        tf.reset_default_graph()
        df_data = perform_cnn_training(training_generator, test_generator)

        q.put(df_data)


df_data = np.ndarray(shape=(0, 6))
df_data_queue = multiprocessing.Queue()
ITER_IN_PROCSS = True
for train_data_percentage in [100, 90, 80, 70, 60, 50, 40]:
    for seed in range(7):
        if ITER_IN_PROCSS:
            p = multiprocessing.Process(target=run_training_step, args=(df_data_queue,))
            p.start()
            res = df_data_queue.get()
            if res is not None:
                df_data = np.append(df_data, res, axis=0)
            p.join()
        else:
            run_training_step(df_data_queue)
            res = df_data_queue.get()
            if res is not None:
                df_data = np.append(df_data, res, axis=0)

        np.savetxt(DBFILE, df_data, delimiter=',', fmt='%f')




import pandas as pd
df = pd.DataFrame(data=df_data,
                  columns=["train_data_percentage", "fold_nr", "run_type", "epoch", "loss", "roc_auc"]) # run_type is either GAN (0), or CNN(1)

# Save the dataframe to disk
import cPickle as pickle
with open("./dataframe_runs", 'wb') as fp:
    pickle.dump(df, fp)



# https://stackoverflow.com/questions/22481854/plot-mean-and-standard-deviation
df_mean = df.groupby(['train_data_percentage', 'run_type'])['roc_auc'].mean()
df_std = df.groupby(['train_data_percentage', 'run_type'])['roc_auc'].std()

df_mean = df_mean.reset_index()
df_std = df_std.reset_index()

gan_rocs_mean = df_mean[df_mean['run_type'] == 0]
gan_rocs_std = df_std[df_std['run_type'] == 0]

cnn_rocs_mean = df_mean[df_mean['run_type'] == 1]
cnn_rocs_std = df_std[df_std['run_type'] == 1]

# df_mean_err, df_std_err = None, None
# if df_mean.hasnans or df_std.hasnans:
#   print "ERROR: df_mean of df_std have nans. Check variable df_mean_err, df_std_err"
#   df_mean_err, df_std_err = df_mean, df_std


plt.errorbar(gan_rocs_mean['train_data_percentage'].values, gan_rocs_mean['roc_auc'].values, gan_rocs_std['roc_auc'].values, marker='^')
plt.errorbar(cnn_rocs_mean['train_data_percentage'].values, cnn_rocs_mean['roc_auc'].values, cnn_rocs_std['roc_auc'].values, marker='*')
plt.savefig("errorbar_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png")

