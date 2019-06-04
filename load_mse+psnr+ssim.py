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
project_root = '.'
model_file = "model_K5_D5C4_TV_iso_e-8_ComplexConv_Kloss_e-8"
model_save_path = join(project_root, 'models/%s' % model_file)
model_name = "model.ckpt"

lambda_num = '1e-8'

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



def evaluate(test_data, mask):
    with tf.Graph() .as_default() as g:
        #x = tf.placeholder(tf.float32, shape=[None, 6, 117, 120, 2], name='x-input')
        y_ = tf.placeholder(tf.float32, shape=[None, 6, 117, 120, 2], name='y-label')
        mask_p = tf.placeholder(tf.complex64, shape=[None, 6, 117, 120], name='mask')
        kspace_p = tf.placeholder(tf.complex64, shape=[None, 6, 117, 120], name='kspace')
        kspace_full = tf.placeholder(tf.complex64, shape=[None, 6, 117, 120], name='kspace_full')

        y, k_recon, block_1, block_2, block_3 = inference.inference(mask_p, kspace_p, None)

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
                            loss_value, y_pred = sess.run([loss, y],
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
                            loss_value, y_pred = sess.run([loss, y],
                                                          feed_dict={y_: ys_l, mask_p: mask_l,
                                                                     kspace_p: kspace_l, kspace_full: k_full_l})
                            time_end = time.time()
                            y_pred_new = real2complex(y_pred)
                            xs = real2complex(xs_l)


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



                        scio.savemat(join(quantization_save_path, 'Test_MSE_%s' % lambda_num), {'test_mse': Test_MSE})
                        scio.savemat(join(quantization_save_path, 'Test_PSNR_%s' % lambda_num), {'test_psnr': Test_PSNR})
                        scio.savemat(join(quantization_save_path, 'Test_SSIM_%s' % lambda_num), {'test_ssim': Test_SSIM})
                        scio.savemat(join(quantization_save_path, 'Base_MSE_%s' % lambda_num), {'base_mse': Base_MSE})
                        scio.savemat(join(quantization_save_path, 'Base_PSNR_%s' % lambda_num), {'base_psnr': Base_PSNR})
                        scio.savemat(join(quantization_save_path, 'Base_SSIM_%s' % lambda_num), {'base_ssim': Base_SSIM})
                            #plt.show()
                        #elif test_case == "Save image":

            else:
                print("No checkpoint file found")


def main(argv=None):
    test_data = Test_data()
    mask = np.load("test_mask.npy")
    acc = mask.size / np.sum(mask)
    print ('Acceleration Rate:{:.2f}'.format(acc))
    """
    mask = cs.cartesian_mask(input_shape, ivar=0.003,
                             centred=False,
                             sample_high_freq=True,
                             sample_centre=True,
                             sample_n=6)

    acc = mask.size / np.sum(mask)
    print ('Acceleration Rate:{:.2f}'.format(acc))
    np.save("test_mask.npy", mask)
    """
    evaluate(test_data, mask)

if __name__ == '__main__':
    tf.app.run()






