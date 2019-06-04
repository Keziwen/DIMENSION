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

                        K_Test_MSE = []
                        K_Test_PSNR = []
                        K_Test_SSIM = []

                        Block1_Test_MSE = []
                        Block1_Test_PSNR = []
                        Block1_Test_SSIM = []

                        Block2_Test_MSE = []
                        Block2_Test_PSNR = []
                        Block2_Test_SSIM = []

                        Block3_Test_MSE = []
                        Block3_Test_PSNR = []
                        Block3_Test_SSIM = []

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
                            k_recon_pred = real2complex(k_recon_pred)
                            block_1_pred = real2complex(block_1_pred)
                            block_2_pred = real2complex(block_2_pred)
                            block_3_pred = real2complex(block_3_pred)
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

                            base_mse, k_test_mse, base_psnr, \
                            k_test_psnr, base_ssim, k_test_ssim = performance(xs, k_recon_pred, ys)

                            base_mse, block1_test_mse, base_psnr, \
                            block1_test_psnr, base_ssim, block1_test_ssim = performance(xs, block_1_pred, ys)

                            base_mse, block2_test_mse, base_psnr, \
                            block2_test_psnr, base_ssim, block2_test_ssim = performance(xs, block_2_pred, ys)

                            base_mse, block3_test_mse, base_psnr, \
                            block3_test_psnr, base_ssim, block3_test_ssim = performance(xs, block_3_pred, ys)

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
                            k_test_mse = ("%.6f" % k_test_mse)
                            block1_test_mse = ("%.6f" % block1_test_mse)
                            block2_test_mse = ("%.6f" % block2_test_mse)
                            block3_test_mse = ("%.6f" % block3_test_mse)

                            Test_MSE.append(test_mse)
                            Test_PSNR.append(test_psnr)
                            Test_SSIM.append(test_ssim)

                            K_Test_MSE.append(k_test_mse)
                            K_Test_PSNR.append(k_test_psnr)
                            K_Test_SSIM.append(k_test_ssim)

                            Block1_Test_MSE.append(block1_test_mse)
                            Block1_Test_PSNR.append(block1_test_psnr)
                            Block1_Test_SSIM.append(block1_test_ssim)

                            Block2_Test_MSE.append(block2_test_mse)
                            Block2_Test_PSNR.append(block2_test_psnr)
                            Block2_Test_SSIM.append(block2_test_ssim)

                            Block3_Test_MSE.append(block3_test_mse)
                            Block3_Test_PSNR.append(block3_test_psnr)
                            Block3_Test_SSIM.append(block3_test_ssim)


                            Base_MSE.append(base_mse)
                            Base_PSNR.append(base_psnr)
                            Base_SSIM.append(base_ssim)


                            mask_shift = mymath.fftshift(mask, axes=(-1, -2))
                            gamma = 1
                            plt.figure(1)
                            plt.subplot(221)
                            plt.imshow(exposure.adjust_gamma(np.abs(ys[0][0]), gamma), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title('ground truth')
                            plt.subplot(222)
                            plt.imshow(exposure.adjust_gamma(abs(mask_shift[0][0]), gamma), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title('mask')
                            plt.subplot(223)
                            plt.imshow(exposure.adjust_gamma(abs(xs[0][0]), gamma), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title("undersampling")
                            plt.subplot(224)
                            plt.imshow(exposure.adjust_gamma(abs(y_pred_new[0][0]), gamma), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title("reconstruction")
                            plt.savefig(join(figure_save_path, 'test%s.tif' % order), dpi=300)

                            plt.figure(2)
                            plt.imshow(exposure.adjust_gamma(np.abs(ys[0][0]), gamma), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title('ground truth')
                            scio.savemat(join(mat_save_path, 'gr%s' % order), {'gr': abs(ys[0][0])})
                            plt.savefig(join(figure_save_path, 'gr%s.tif' % order), dpi=300)

                            plt.figure(3)
                            plt.imshow(exposure.adjust_gamma(abs(xs[0][0]), gamma), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title("undersampling: " + base_mse + ' ' + str(round(base_psnr, 5)) + ' ' + str(
                                round(base_ssim, 4)))
                            scio.savemat(join(mat_save_path, 'under%s' % order), {'under': abs(xs[0][0])})
                            plt.savefig(join(figure_save_path, 'under%s.tif' % order), dpi=300)

                            plt.figure(4)
                            plt.imshow(exposure.adjust_gamma(abs(y_pred_new[0][0]), gamma), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title("reconstruction: " + test_mse + ' ' + str(round(test_psnr, 5)) + ' ' + str(
                                round(test_ssim, 4)))
                            scio.savemat(join(mat_save_path, 'recon%s' % order), {'recon': abs(y_pred_new[0][0])})
                            plt.savefig(join(figure_save_path, 'recon%s.tif' % order), dpi=300)

                            plt.figure(5)
                            plt.imshow(exposure.adjust_gamma(abs(k_recon_pred[0][0]), gamma), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title("k_recon: " + k_test_mse + ' ' + str(round(k_test_psnr, 5)) + ' ' + str(
                                round(k_test_ssim, 4)))
                            scio.savemat(join(mat_save_path, 'k_recon%s' % order), {'k_recon': abs(k_recon_pred[0][0])})
                            plt.savefig(join(figure_save_path, 'k_recon%s.tif' % order), dpi=300)

                            plt.figure(6)
                            plt.imshow(exposure.adjust_gamma(abs(block_1_pred[0][0]), gamma), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title("block1_recon: " + block1_test_mse + ' ' + str(round(block1_test_psnr, 5)) + ' ' + str(
                                round(block1_test_ssim, 4)))
                            scio.savemat(join(mat_save_path, 'block1_recon%s' % order), {'block1_recon': abs(block_1_pred[0][0])})
                            plt.savefig(join(figure_save_path, 'block1_recon%s.tif' % order), dpi=300)

                            plt.figure(7)
                            plt.imshow(exposure.adjust_gamma(abs(block_2_pred[0][0]), gamma), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title(
                                "block2_recon: " + block2_test_mse + ' ' + str(round(block2_test_psnr, 5)) + ' ' + str(
                                    round(block2_test_ssim, 4)))
                            scio.savemat(join(mat_save_path, 'block2_recon%s' % order),
                                         {'block2_recon': abs(block_2_pred[0][0])})
                            plt.savefig(join(figure_save_path, 'block2_recon%s.tif' % order), dpi=300)

                            plt.figure(8)
                            plt.imshow(exposure.adjust_gamma(abs(block_3_pred[0][0]), gamma), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title(
                                "block3_recon: " + block3_test_mse+ ' ' + str(round(block3_test_psnr, 5)) + ' ' + str(
                                    round(block3_test_ssim, 4)))
                            scio.savemat(join(mat_save_path, 'block3_recon%s' % order),
                                         {'block3_recon': abs(block_3_pred[0][0])})
                            plt.savefig(join(figure_save_path, 'block3_recon%s.tif' % order), dpi=300)

                            plt.figure(9)
                            plt.imshow(exposure.adjust_gamma(abs(abs(ys[0][0]) - abs(y_pred_new[0][0])), gamma), vmin=0,
                                       vmax=0.07)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title("error: " + test_mse + ' ' + str(round(test_psnr, 5)) + ' ' + str(
                                round(test_ssim, 4)))
                            scio.savemat(join(mat_save_path, 'error%s' % order),
                                         {'error': abs(abs(ys[0][0]) - abs(y_pred_new[0][0]))})
                            plt.savefig(join(figure_save_path, 'error%s.tif' % order), dpi=300)

                            plt.figure(10)
                            plt.imshow(exposure.adjust_gamma(abs(abs(ys[0][0]) - abs(k_recon_pred[0][0])), gamma), vmin=0,
                                       vmax=0.07)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title("k_error: " + k_test_mse+ ' ' + str(round(k_test_psnr, 5)) + ' ' + str(
                                round(k_test_ssim, 4)))
                            scio.savemat(join(mat_save_path, 'k_error%s' % order),
                                         {'k_error': abs(abs(ys[0][0]) - abs(k_recon_pred[0][0]))})
                            plt.savefig(join(figure_save_path, 'k_error%s.tif' % order), dpi=300)

                            plt.figure(11)
                            plt.imshow(exposure.adjust_gamma(abs(abs(ys[0][0]) - abs(block_1_pred[0][0])), gamma),
                                       vmin=0,
                                       vmax=0.07)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title(
                                "block1_error: " + block1_test_mse+ ' ' + str(round(block1_test_psnr, 5)) + ' ' + str(
                                    round(block1_test_ssim, 4)))
                            scio.savemat(join(mat_save_path, 'block1_error%s' % order),
                                         {'block1_error': abs(abs(ys[0][0]) - abs(block_1_pred[0][0]))})
                            plt.savefig(join(figure_save_path, 'block1_error%s.tif' % order), dpi=300)

                            plt.figure(12)
                            plt.imshow(exposure.adjust_gamma(abs(abs(ys[0][0]) - abs(block_2_pred[0][0])), gamma),
                                       vmin=0,
                                       vmax=0.07)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title(
                                "block2_error: " + block2_test_mse + ' ' + str(
                                    round(block2_test_psnr, 5)) + ' ' + str(
                                    round(block2_test_ssim, 4)))
                            scio.savemat(join(mat_save_path, 'block2_error%s' % order),
                                         {'block2_error': abs(abs(ys[0][0]) - abs(block_2_pred[0][0]))})
                            plt.savefig(join(figure_save_path, 'block2_error%s.tif' % order), dpi=300)

                            plt.figure(13)
                            plt.imshow(exposure.adjust_gamma(abs(abs(ys[0][0]) - abs(block_3_pred[0][0])), gamma),
                                       vmin=0,
                                       vmax=0.07)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title(
                                "block3_error: " + block3_test_mse + ' ' + str(
                                    round(block3_test_psnr, 5)) + ' ' + str(
                                    round(block3_test_ssim, 4)))
                            scio.savemat(join(mat_save_path, 'block3_error%s' % order),
                                         {'block3_error': abs(abs(ys[0][0]) - abs(block_3_pred[0][0]))})
                            plt.savefig(join(figure_save_path, 'block3_error%s.tif' % order), dpi=300)

                            plt.figure(14)
                            plt.subplot(511)
                            plt.imshow(np.abs(ys_t[0]), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title("gnd_t_y")
                            plt.subplot(512)
                            plt.imshow(np.abs(xs_t[0]), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title("under_t_y")
                            plt.subplot(513)
                            plt.imshow(np.abs(xs_t_error[0]))
                            plt.xticks([])
                            plt.yticks([])
                            plt.title("under_t_y_error")
                            plt.subplot(514)
                            plt.imshow(np.abs(y_pred_t[0]), plt.cm.gray)
                            plt.xticks([])
                            plt.yticks([])
                            plt.title("recon_t_y")
                            plt.subplot(515)
                            plt.imshow(np.abs(y_pred_error[0]))
                            plt.xticks([])
                            plt.yticks([])
                            plt.title("recon_t_y_error")
                            plt.savefig(join(figure_save_path, 't_y%s.tif' % order))
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

                        scio.savemat(join(quantization_save_path, 'K_Test_MSE'), {'k_test_mse': K_Test_MSE})
                        scio.savemat(join(quantization_save_path, 'K_Test_PSNR'),
                                     {'k_test_psnr': K_Test_PSNR})
                        scio.savemat(join(quantization_save_path, 'K_Test_SSIM'),
                                     {'k_test_ssim': K_Test_SSIM})

                        scio.savemat(join(quantization_save_path, 'Block1_Test_MSE'), {'block1_test_mse': Block1_Test_MSE})
                        scio.savemat(join(quantization_save_path, 'Block1_Test_PSNR'),
                                     {'block1_test_psnr': Block1_Test_PSNR})
                        scio.savemat(join(quantization_save_path, 'Block1_Test_SSIM'),
                                     {'block1_test_ssim': Block1_Test_SSIM})

                        scio.savemat(join(quantization_save_path, 'Block2_Test_MSE'),
                                     {'block2_test_mse': Block2_Test_MSE})
                        scio.savemat(join(quantization_save_path, 'Block2_Test_PSNR'),
                                     {'block2_test_psnr': Block2_Test_PSNR})
                        scio.savemat(join(quantization_save_path, 'Block2_Test_SSIM'),
                                     {'block2_test_ssim': Block2_Test_SSIM})

                        scio.savemat(join(quantization_save_path, 'Block3_Test_MSE'),
                                     {'block3_test_mse': Block3_Test_MSE})
                        scio.savemat(join(quantization_save_path, 'Block3_Test_PSNR'),
                                     {'block3_test_psnr': Block3_Test_PSNR})
                        scio.savemat(join(quantization_save_path, 'Block3_Test_SSIM'),
                                     {'block3_test_ssim': Block3_Test_SSIM})

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

    model_file = "model_K5_D5C4_TV_iso_e-8_ComplexConv"
    model = join(project_root, 'models/%s' % model_file)
    model_name = "model.ckpt"
    evaluate(test_data, mask, model_save_path=model, model_file=model_file)


if __name__ == '__main__':
    tf.app.run()






