import time
import tensorflow as tf
import numpy as np
import compressed_sensing as cs
import matplotlib.pyplot as plt
from skimage import exposure
import mymath
import os
from os.path import join
import scipy.io as scio


from utils.metric import complex_psnr
from utils.metric import mse
from skimage.measure import compare_ssim

import inference
import train

from datasets import Test_data


batch_size = 1
input_shape = [batch_size, 6, 117, 120]
model_name = "model.ckpt"


def real2complex(x):
    '''
        Converts from array of the form ([n, ]nt, nx, ny 2) to ([n, ] nt, nx, ny)
        '''
    x = np.asarray(x)
    if x.shape[0] == 2 and x.shape[1] != 2:  # Hacky check
        return x[0] + x[1] * 1j
    elif x.shape[4] == 2:
        y = x[:, :, :, :,  0] + x[:, :, :, :, 1] * 1j
        return y
    else:
        raise ValueError('Invalid dimension')

def performance(xs, y, ys):
    base_mse = mse(ys, xs)
    test_mse = mse(ys, y)
    base_psnr = complex_psnr(ys, xs, peak='max')
    test_psnr = complex_psnr(ys, y, peak='max')
    batch, nt, nx, ny = y.shape
    base_ssim = 0
    test_ssim = 0
    for i in range(nt):
        base_ssim += compare_ssim(np.abs(ys[0][i]).astype('float64'),
                                  np.abs(xs[0][i]).astype('float64'))
        test_ssim += compare_ssim(np.abs(ys[0][i]).astype('float64'),
                                  np.abs(y[0][i]).astype('float64'))
    base_ssim /= nt
    test_ssim /= nt
    return base_mse, test_mse, base_psnr, test_psnr, base_ssim, test_ssim



def evaluate(test_data, mask, model_save_path, model_file):
    with tf.Graph() .as_default() as g:
        #x = tf.placeholder(tf.float32, shape=[None, 6, 117, 120, 2], name='x-input')
        y_ = tf.placeholder(tf.float32, shape=[None, 6, 117, 120, 2], name='y-label')
        mask_p = tf.placeholder(tf.complex64, shape=[None, 6, 117, 120], name='mask')
        kspace_p = tf.placeholder(tf.complex64, shape=[None, 6, 117, 120], name='kspace')
        kspace_full = tf.placeholder(tf.complex64, shape=[None, 6, 117, 120], name='kspace_full')

        y, block_1, block_2, block_3, block_k_1 = inference.inference(mask_p, kspace_p, None)

        loss = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(y, y_), [1, 2, 3, 4]))

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(model_save_path)
            saver = tf.train.Saver()
            test_case = 'show image'
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                if __name__ == '__main__':
                    if test_case == 'check_loss':
                        count = 0
                        for ys in train.iterate_minibatch(test_data, batch_size, shuffle=True):
                            xs_l, kspace_l, mask_l, ys_l, k_full_l = train.prep_input(test_data, mask)
                            loss_value, y_pred, block_1_pred, block_2_pred, block_3_pred, block_k_pred = sess.run([loss, y, block_1, block_2, block_3, block_k_1],
                                                          feed_dict={y_: ys_l, mask_p: mask_l, kspace_p: kspace_l, kspace_full: k_full_l})
                            print("The loss of No.{} test data = {}".format(count + 1, loss_value))

                            y_c = real2complex(y_pred)
                            xs_c = real2complex(xs_l)
                            base_mse, test_mse, base_psnr, \
                            test_psnr, base_ssim, test_ssim = performance(xs_c, y_c, ys)
                            print("test loss:\t\t{:.6f}".format(loss_value))
                            print("test psnr:\t\t{:.6f}".format(test_psnr))
                            print("base psnr:\t\t{:.6f}".format(base_psnr))
                            print("base mse:\t\t{:.6f}".format(base_mse))
                            print("test mse:\t\t{:.6f}".format(test_mse))
                            print("base ssim:\t\t{:.6f}".format(base_ssim))
                            print("test ssim:\t\t{:.6f}".format(test_ssim))
                            count += 1
                    elif test_case == 'show image':
                        project_root = '.'

                        figure_save_path = join(project_root, 'result/images/%s' % model_file)
                        if not os.path.isdir(figure_save_path):
                            os.makedirs(figure_save_path)

                        mat_save_path = join(project_root, 'result/mat/%s' % model_file)
                        if not os.path.isdir(mat_save_path):
                            os.makedirs(mat_save_path)

                        quantization_save_path = join(project_root, 'result/quantization/%s' % model_file)
                        if not os.path.isdir(quantization_save_path):
                            os.makedirs(quantization_save_path)

                        Test_MSE = []
                        Test_PSNR = []
                        Test_SSIM = []

                        Base_MSE = []
                        Base_PSNR = []
                        Base_SSIM = []

                        for order in range(0, 100):
                            ys = test_data[order]
                            ys = ys[np.newaxis, :]
                            xs_l, kspace_l, mask_l, ys_l, k_full_l = train.prep_input(ys, mask)
                            time_start = time.time()
                            loss_value, y_pred, block_1_pred, block_2_pred, block_3_pred, k_recon_pred = sess.run([loss, y, block_1, block_2, block_3, block_k_1],
                                                          feed_dict={y_: ys_l, mask_p: mask_l,
                                                                     kspace_p: kspace_l, kspace_full: k_full_l})
                            time_end = time.time()
                            y_pred_new = real2complex(y_pred)
                            xs = real2complex(xs_l)
                            if order == 0:
                                order_x = 100
                            elif order == 1:
                                order_x = 60
                            elif order == 2:
                                order_x = 85
                            elif order == 6:
                                order_x = 40
                            else:
                                order_x = 55
                            # order_x = 55 # (order, order_x): (0, 100), (1, 60), (6, 40), (7, 55)
                            ys_t = ys[:, :, order_x, :]
                            y_pred_t = y_pred_new[:, :, order_x, :]
                            xs_t = xs[:, :, order_x, :]
                            xs_t_error = ys_t - xs_t
                            y_pred_error = ys_t - y_pred_t

                            base_mse, test_mse, base_psnr, \
                            test_psnr, base_ssim, test_ssim = performance(xs, y_pred_new, ys)


                            print("test time:\t\t{:.6f}".format(time_end - time_start))
                            print("test loss:\t\t{:.6f}".format(loss_value))
                            print("test psnr:\t\t{:.6f}".format(test_psnr))
                            print("base psnr:\t\t{:.6f}".format(base_psnr))
                            print("base mse:\t\t{:.6f}".format(base_mse))
                            print("test mse:\t\t{:.6f}".format(test_mse))
                            print("base ssim:\t\t{:.6f}".format(base_ssim))
                            print("test ssim:\t\t{:.6f}".format(test_ssim))
                            base_mse = ("%.6f" % base_mse)
                            test_mse = ("%.6f" % test_mse)

                            Test_MSE.append(test_mse)
                            Test_PSNR.append(test_psnr)
                            Test_SSIM.append(test_ssim)

                            Base_MSE.append(base_mse)
                            Base_PSNR.append(base_psnr)
                            Base_SSIM.append(base_ssim)

                            train_plot = np.load(join(project_root, 'models/%s' % model_file, 'train_plot.npy'))
                            validate_plot = np.load(join(project_root, 'models/%s' % model_file, 'validate_plot.npy'))
                            [num_train_plot, ] = train_plot.shape
                            [num_validate_plot, ] = validate_plot.shape
                            x1 = np.arange(1, num_train_plot + 1)
                            x2 = np.arange(1, num_validate_plot + 1)

                            plt.figure(15)
                            l1, = plt.plot(x1, train_plot)
                            l2, = plt.plot(x2, validate_plot)
                            plt.legend(handles=[l1, l2, ], labels=['train loss', 'validation loss'], loc=1)
                            plt.xlabel('epoch')
                            plt.ylabel('loss')
                            plt.title('loss')
                            if not os.path.exists(join(figure_save_path, 'loss.tif')):
                                plt.savefig(join(figure_save_path, 'loss.tif'), dpi=300)
                            #plt.show()

                        scio.savemat(join(quantization_save_path, 'Test_MSE'), {'test_mse': Test_MSE})
                        scio.savemat(join(quantization_save_path, 'Test_PSNR'),
                                     {'test_psnr': Test_PSNR})
                        scio.savemat(join(quantization_save_path, 'Test_SSIM'),
                                     {'test_ssim': Test_SSIM})


                        scio.savemat(join(quantization_save_path, 'Base_MSE'), {'base_mse': Base_MSE})
                        scio.savemat(join(quantization_save_path, 'Base_PSNR'),
                                     {'base_psnr': Base_PSNR})
                        scio.savemat(join(quantization_save_path, 'Base_SSIM'),
                                     {'base_ssim': Base_SSIM})
                        #elif test_case == "Save image":

            else:
                print("No checkpoint file found")


def main(argv=None):
    test_data = Test_data()
    mask = np.load("test_mask.npy")
    acc = mask.size / np.sum(mask)
    print('Acceleration Rate:{:.2f}'.format(acc))

    project_root = '.'
    model_file = "model_K5_D9C4_TV_iso_e-8_ComplexConv_kLoss_e-1_block1-3_e+3_e+3_e+3"
    model = join(project_root, 'models/%s' % model_file)
    model_name = "model.ckpt"
    evaluate(test_data, mask, model_save_path=model, model_file=model_file)


if __name__ == '__main__':
    tf.app.run()






