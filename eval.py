import time
import tensorflow as tf
import numpy as np
import compressed_sensing as cs
import matplotlib.pyplot as plt
from skimage import exposure
import mymath
import os
from os.path import join


from utils.metric import complex_psnr
from utils.metric import mse
from skimage.measure import compare_ssim

import inference
import train

from datasets import Test_data


batch_size = 1
input_shape = [batch_size, 6, 117, 120]
project_root = '.'
model_file = "model_KI_TV_iso_e-8"
model_save_path = join(project_root, 'models/%s' % model_file)
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



def evaluate(test_data, mask):
    with tf.Graph() .as_default() as g:
        y_ = tf.placeholder(tf.float32, shape=[None, 6, 117, 120, 2], name='y-label')
        mask_p = tf.placeholder(tf.complex64, shape=[None, 6, 117, 120], name='mask')
        kspace_p = tf.placeholder(tf.complex64, shape=[None, 6, 117, 120], name='kspace')

        y = inference.inference(mask_p, kspace_p, None)

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
                            xs_l, kspace_l, mask_l, ys_l = train.prep_input(ys, mask)
                            loss_value, y_pred = sess.run([loss, y],
                                                          feed_dict={y_: ys_l, mask_p: mask_l, kspace_p: kspace_l})
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
                        final_result_path = '/media/keziwen/86AA9651AA963E1D/Tensorflow/MyDeepMRI-KI_Net V2/for KI/final results'
                        figure_save_path = join(final_result_path, 'KI vs KI_with_KLoss(e-1)', 'KI')
                        if not os.path.isdir(figure_save_path):
                            os.makedirs(figure_save_path)
                        order = 79
                        ys = test_data[order]
                        ys = ys[np.newaxis, :]
                        xs_l, kspace_l, mask_l, ys_l = train.prep_input(ys, mask)
                        time_start = time.time()
                        loss_value, y_pred = sess.run([loss, y],
                                                 feed_dict={y_: ys_l, mask_p: mask_l, kspace_p: kspace_l})
                        time_end = time.time()
                        y_pred = real2complex(y_pred)
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
                        #order_x = 55 # (order, order_x): (0, 100), (1, 60), (6, 40), (7, 55)
                        ys_t = ys[:, :, order_x, :]
                        y_pred_t = y_pred[:, :, order_x, :]
                        xs_t = xs[:, :, order_x, :]
                        xs_t_error = ys_t - xs_t
                        y_pred_error = ys_t - y_pred_t

                        base_mse, test_mse, base_psnr,\
                        test_psnr, base_ssim, test_ssim = performance(xs, y_pred, ys)
                        print("test time:\t\t{:.6f}".format(time_end-time_start))
                        print("test loss:\t\t{:.6f}".format(loss_value))
                        print("test psnr:\t\t{:.6f}".format(test_psnr))
                        print("base psnr:\t\t{:.6f}".format(base_psnr))
                        print("base mse:\t\t{:.6f}".format(base_mse))
                        print("test mse:\t\t{:.6f}".format(test_mse))
                        print("base ssim:\t\t{:.6f}".format(base_ssim))
                        print("test ssim:\t\t{:.6f}".format(test_ssim))
                        mask_shift = mymath.fftshift(mask, axes=(-1, -2))
                        gamma = 1
                        plt.figure(1)
                        plt.subplot(221)
                        plt.imshow(exposure.adjust_gamma(np.abs(ys[0][0]), gamma), plt.cm.gray)
                        plt.title('ground truth')
                        plt.subplot(222)
                        plt.imshow(exposure.adjust_gamma(abs(mask_shift[0][0]), gamma), plt.cm.gray)
                        plt.title('mask')
                        plt.subplot(223)
                        plt.imshow(exposure.adjust_gamma(abs(xs[0][0]), gamma), plt.cm.gray)
                        plt.title("undersampling")
                        plt.subplot(224)
                        plt.imshow(exposure.adjust_gamma(abs(y_pred[0][0]), gamma), plt.cm.gray)
                        plt.title("reconstruction")
                        plt.savefig(join(figure_save_path, 'test%s.png' % order))
                        plt.figure(2)
                        plt.imshow(exposure.adjust_gamma(np.abs(ys[0][0]), gamma), plt.cm.gray)
                        plt.title('ground truth')
                        plt.savefig(join(figure_save_path, 'gr%s.png' % order))
                        plt.figure(3)
                        plt.imshow(exposure.adjust_gamma(abs(xs[0][0]), gamma), plt.cm.gray)
                        plt.title("undersampling")
                        plt.savefig(join(figure_save_path, 'under%s.png' % order))
                        plt.figure(4)
                        plt.imshow(exposure.adjust_gamma(abs(y_pred[0][0]), gamma), plt.cm.gray)
                        plt.title("reconstruction")
                        plt.savefig(join(figure_save_path, 'recon%s.png' % order))
                        plt.figure(5)
                        plt.imshow(exposure.adjust_gamma(abs(np.abs(ys[0][0]) - abs(y_pred[0][0])), gamma))
                        plt.title("error")
                        plt.savefig(join(figure_save_path, 'error%s.png' % order))
                        plt.figure(6)
                        plt.subplot(511)
                        plt.imshow(np.abs(ys_t[0]), plt.cm.gray)
                        plt.title("gnd_t_y")
                        plt.subplot(512)
                        plt.imshow(np.abs(xs_t[0]), plt.cm.gray)
                        plt.title("under_t_y")
                        plt.subplot(513)
                        plt.imshow(np.abs(xs_t_error[0]))
                        plt.title("under_t_y_error")
                        plt.subplot(514)
                        plt.imshow(np.abs(y_pred_t[0]), plt.cm.gray)
                        plt.title("recon_t_y")
                        plt.subplot(515)
                        plt.imshow(np.abs(y_pred_error[0]))
                        plt.title("recon_t_y_error")
                        plt.savefig(join(figure_save_path, 't_y%s.png' % order))
                        train_plot = np.load(join(project_root, 'models/%s' % model_file, 'train_plot.npy'))
                        validate_plot = np.load(join(project_root, 'models/%s' % model_file, 'validate_plot.npy'))
                        [num_train_plot, ] = train_plot.shape
                        [num_validate_plot, ] = validate_plot.shape
                        x1 = np.arange(1, num_train_plot + 1)
                        x2 = np.arange(1, num_validate_plot + 1)
                        plt.figure(7)
                        l1, = plt.plot(x1, train_plot)
                        l2, = plt.plot(x2, validate_plot)
                        plt.legend(handles=[l1, l2, ], labels=['train loss', 'validation loss'], loc=1)
                        plt.xlabel('epoch')
                        plt.ylabel('loss')
                        plt.title('loss')
                        if not os.path.exists(join(figure_save_path, 'loss.png')):
                            plt.savefig(join(figure_save_path, 'loss.png'))
                        #plt.show()
                    #elif test_case == "Save image":




            else:
                print("No checkpoint file found")


def main(argv=None):
    test_data = Test_data()
    mask = np.load("test_mask.npy")
    acc = mask.size / np.sum(mask)
    print ('Acceleration Rate:{:.2f}'.format(acc))
    evaluate(test_data, mask)

if __name__ == '__main__':
    tf.app.run()






