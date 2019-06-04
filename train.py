import tensorflow as tf
import numpy as np
import inference
import compressed_sensing as cs
from helpers import to_lasagne_format
import time
from os.path import join
import os

from datasets import Train_data

batch_size = 10
learning_rate_base = 0.0001
learning_rate_decay = 0.95
regularization_rate = 1e-7
Train_steps = 30
num_train = 15000
num_validate = 2000
input_shape = [batch_size, 6, 117, 120]

project_root = '.'
model_file = "model_K5_D3C4_TV_iso_e-8_ComplexConv_kLoss_e-1_block1-3_e+3_e+3_e+3"
model_save_path = join(project_root, 'models/%s' % model_file)
if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)
model_name = "model.ckpt"


def iterate_minibatch(data, batch_size, shuffle=False):
    n = len(data)

    if shuffle:
        data = np.random.permutation(data)

    for i in xrange(0, n, batch_size):
        yield data[i:i+batch_size]


def prep_input(ys, mask):
    """Undersample the batch, then reformat them into what the network accepts.

    Parameters
    ----------
    gauss_ivar: float - controls the undersampling rate.
                        higher the value, more undersampling
    """
    """
    mask = cs.cartesian_mask(ys.shape, gauss_ivar,
                             centred=False,
                             sample_high_freq=True,
                             sample_centre=True,
                             sample_n=6)
    """
    xs, k_und, k_full = cs.undersample(ys, mask, centred=False, norm=None)

    ys_l = to_lasagne_format(ys)
    xs_l = to_lasagne_format(xs)
    mask = mask.astype(np.complex)
    return xs_l, k_und, mask, ys_l, k_full

def TV(f, case=1):
    indices_x = np.random.randint(1, 116, [117])
    indices_x[0:116] = range(1, 117)
    indices_x[116] = 0
    indices_y = np.random.randint(1, 119, [120])
    indices_y[0:119] = range(1, 120)
    indices_y[119] = 0

    f_x = tf.gather(f, indices=indices_x, axis=2) - f
    f_y = tf.gather(f, indices=indices_y, axis=3) - f

    # anisotropy
    if case == 1:
        TV_f = tf.reduce_mean(tf.reduce_sum(tf.abs(f_x) + tf.abs(f_y), [1, 2, 3, 4]))
        print("Using anisotropy TV")

    # isotropy
    if case == 2:
        TV_f = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(f_x) + tf.square(f_y)), [1, 2, 3, 4]))
        print("Using isotropy TV")
    return TV_f

def HDTV(f, case=1, degree=2):
    indices_x = np.random.randint(1, 116, [117])
    indices_x[0:116] = range(1, 117)
    indices_x[116] = 0
    indices_y = np.random.randint(1, 119, [120])
    indices_y[0:119] = range(1, 120)
    indices_y[119] = 0

    f_x = tf.gather(f, indices=indices_x, axis=2) - f
    f_y = tf.gather(f, indices=indices_y, axis=3) - f

    if degree == 2:
        f_xx_n = tf.gather(f_x, indices=indices_x, axis=2) - f_x
        f_yy_n = tf.gather(f_y, indices=indices_y, axis=3) - f_y
        f_xy_n = tf.gather(f_x, indices=indices_y, axis=3) - f_x

    if degree == 3:
        f_xx = tf.gather(f_x, indices=indices_x, axis=2) - f_x
        f_yy = tf.gather(f_y, indices=indices_y, axis=3) - f_y
        f_xy = tf.gather(f_x, indices=indices_y, axis=3) - f_x

        f_xx_n = tf.gather(f_xx, indices=indices_x, axis=2) - f_xx
        f_yy_n = tf.gather(f_yy, indices=indices_y, axis=3) - f_yy
        f_xxy_n = tf.gather(f_xx, indices=indices_y, axis=3) - f_xx
        f_xyy_n = tf.gather(f_xy, indices=indices_y, axis=3) - f_xy

    if case == 1:
        if degree == 2:
            HDTV_f = tf.reduce_mean(tf.reduce_sum(tf.sqrt((3 * tf.square(f_xx_n) + 3 * tf.square(f_yy_n)
                                                          + 4 * tf.square(f_xy_n) + tf.multiply(f_xx_n, f_yy_n))
                                                          / 8), [1, 2, 3, 4]))
        if degree == 3:
            HDTV_f = tf.reduce_mean(tf.reduce_sum(tf.sqrt((5 * (tf.square(f_xx_n) + tf.square(f_yy_n)) +
                                                           3 * (tf.multiply(f_xx_n, f_xyy_n) + tf.multiply(f_yy_n, f_xxy_n)) +
                                                           9 * (tf.square(f_xxy_n) + tf.square(f_xyy_n)))
                                                          / (4 * np.sqrt(2))),
                                                  [1, 2, 3, 4]))
    return HDTV_f

def real2complex(input_op, inv=False):
    if inv == False:
        return tf.complex(input_op[:, :, :, :, 0], input_op[:, :, :, :, 1])
    else:
        input_real = tf.cast(tf.real(input_op), dtype=tf.float32)
        input_imag = tf.cast(tf.imag(input_op), dtype=tf.float32)
        return tf.stack([input_real, input_imag], axis=4)

def total_loss(y, y_, block_1, block_2, block_3, block_k_1, k_full):
    lambda_TV = 1e-8
    lambda_mse_image2 = 1e+3
    lambda_kspace1 = 1e-1


    loss_mse_image = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(y, y_), [1, 2, 3, 4]))
    loss_mse_image1 = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(block_1, y_), [1, 2, 3, 4]))
    loss_mse_image2 = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(block_2, y_), [1, 2, 3, 4]))
    loss_mse_image3 = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(block_3, y_), [1, 2, 3, 4]))
    #loss_mse_image4 = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(block_4, y_), [1, 2, 3, 4]))

    kspace_full_real = real2complex(k_full, inv=True)
    loss_mse_kspace1 = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(block_k_1, kspace_full_real), [1, 2, 3, 4]))


    loss_TV = TV(y, case=2)
    loss = loss_mse_image + lambda_TV * loss_TV + lambda_kspace1 * loss_mse_kspace1 + lambda_mse_image2 * loss_mse_image1 +\
           lambda_mse_image2 * loss_mse_image2 + lambda_mse_image2 * loss_mse_image3
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'))

def train(train_data, validate_data, mask):
    print ("compling...")
    train_plot = []
    validate_plot = []

    #x = tf.placeholder(tf.float32, shape=[None, 6, 117, 120, 2], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, 6, 117, 120, 2], name='y-label')
    mask_p = tf.placeholder(tf.complex64, shape=[None, 6, 117, 120], name='mask')
    kspace_p = tf.placeholder(tf.complex64, shape=[None, 6, 117, 120], name='kspace')
    kspace_full = tf.placeholder(tf.complex64, shape=[None, 6, 117, 120], name='kspace_full')



    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)

    y, block_1, block_2, block_3, block_k_1 = inference.inference(mask_p, kspace_p, regularizer)

    global_step = tf.Variable(0, trainable=False)

    loss = total_loss(y, y_, block_1, block_2, block_3, block_k_1, kspace_full)

    learning_rate = tf.train.exponential_decay(learning_rate_base,
                                               global_step=global_step,
                                               decay_steps=num_train / batch_size,
                                               decay_rate=learning_rate_decay)
    train_step = tf.train.AdamOptimizer(learning_rate).\
        minimize(loss, global_step=global_step)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_data_per_num = 5000

        # get Initalized value of loss
        count_train = 0
        loss_sum_train = 0.0
        for ys_train in iterate_minibatch(train_data, batch_size, shuffle=True):
            _, kspace_l, mask_l, ys_l, k_full_l = prep_input(ys_train, mask)
            im_start = time.time()
            loss_value_train = sess.run(loss, feed_dict={y_: ys_l,
                                                             mask_p: mask_l, kspace_p: kspace_l, kspace_full: k_full_l
                                                         })
            im_end = time.time()
            loss_sum_train += loss_value_train
            count_train += 1
            print("{}\{} of train loss (just get loss):\t\t{:.6f} \t using :{:.4f}s"
                  .format(count_train, int(num_train / batch_size),
                          loss_sum_train / count_train, im_end - im_start))

        count_validate = 0
        loss_sum_validate = 0.0
        for ys_validate in iterate_minibatch(validate_data, batch_size, shuffle=True):
            _, kspace_l, mask_l, ys_l, k_full_l = prep_input(ys_validate, mask)
            im_start = time.time()
            loss_value_validate = sess.run(loss,
                                           feed_dict={y_: ys_l,
                                                      mask_p: mask_l, kspace_p: kspace_l, kspace_full: k_full_l})
            im_end = time.time()
            loss_sum_validate += loss_value_validate
            count_validate += 1
            print("{}\{} of validation loss:\t\t{:.6f} \t using :{:.4f}s".
                  format(count_validate, int(num_validate / batch_size),
                         loss_sum_validate / count_validate, im_end - im_start))

        train_plot.append(loss_sum_train / count_train)
        validate_plot.append(loss_sum_validate / count_validate)


        for i in range(Train_steps):
            j = 0
            for train_data_per in iterate_minibatch(train_data, batch_size=train_data_per_num, shuffle=True):
                count_train = 0
                loss_sum_train = 0.0
                for ys in iterate_minibatch(train_data_per, batch_size, shuffle=False):
                    _, kspace_l, mask_l, ys_l, k_full_l = prep_input(ys, mask)
                    im_start = time.time()
                    _, loss_value, step = sess.run([train_step, loss, global_step],
                                                   feed_dict={y_: ys_l, mask_p: mask_l, kspace_p: kspace_l, kspace_full: k_full_l})
                    im_end = time.time()
                    loss_sum_train += loss_value
                    print("{}\{}\{}\{} of training loss:\t\t{:.6f} \t using :{:.4f}s".
                          format(i+1, j+1, count_train + 1, int(train_data_per_num / batch_size),
                                 loss_sum_train / (count_train + 1), im_end - im_start))
                    count_train += 1

                # validating and get train loss
                count_train_per = 0
                loss_sum_train_per = 0.0
                for ys_train in iterate_minibatch(train_data_per, batch_size, shuffle=True):
                    _, kspace_l, mask_l, ys_l, k_full_l = prep_input(ys_train, mask)
                    im_start = time.time()
                    loss_value_train_per = sess.run(loss, feed_dict={y_: ys_l,
                                                                     mask_p: mask_l, kspace_p: kspace_l, kspace_full: k_full_l})
                    im_end = time.time()
                    loss_sum_train_per += loss_value_train_per
                    count_train_per += 1
                    print("{}\{}\{}\{} of train loss (just get loss):\t\t{:.6f} \t using :{:.4f}s"
                          .format(i+1, j+1, count_train_per, int(train_data_per_num / batch_size),
                                  loss_sum_train_per / count_train_per, im_end - im_start))

                count_validate = 0
                loss_sum_validate = 0.0
                for ys_validate in iterate_minibatch(validate_data, batch_size, shuffle=True):
                    _, kspace_l, mask_l, ys_l, k_full_l = prep_input(ys_validate, mask)
                    im_start = time.time()
                    loss_value_validate = sess.run(loss,
                                                   feed_dict={y_: ys_l,
                                                              mask_p: mask_l, kspace_p: kspace_l, kspace_full: k_full_l})
                    im_end = time.time()
                    loss_sum_validate += loss_value_validate
                    count_validate += 1
                    print("{}\{}\{}\{} of validation loss:\t\t{:.6f} \t using :{:.4f}s".
                          format(i+1, j+1, count_validate, int(num_validate / batch_size), loss_sum_validate / count_validate, im_end - im_start))

                train_plot.append(loss_sum_train_per / count_train_per)
                validate_plot.append(loss_sum_validate / count_validate)
                j += 1
            print ("After %d train epochs, loss on training batch is %g\n model has been saved in %s"
                   % (i, loss_sum_train / count_train, model_save_path))
            saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)

        train_plot_name = 'train_plot.npy'
        np.save(join(model_save_path, train_plot_name), train_plot)
        validate_plot_name = 'validate_plot.npy'
        np.save(join(model_save_path, validate_plot_name), validate_plot)

def main(argv=None):
    # ivar = 0.003, acc = 4
    # ivar = 0.008, acc = 6
    # ivar = 0.015, acc = 8
    # ivar = 0.030, acc = 10
    # ivar = 0.070, acc = 12
    mask = cs.cartesian_mask(input_shape, ivar=0.003,
                             centred=False,
                             sample_high_freq=True,
                             sample_centre=True,
                             sample_n=6)

    acc = mask.size / np.sum(mask)
    print ('Acceleration Rate:{:.2f}'.format(acc))
    train_data, validate_data = Train_data()
    train(train_data, validate_data, mask)

if __name__ == '__main__':
    tf.app.run()










