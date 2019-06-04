from os.path import join
import numpy as np
import time
import h5py
import tensorflow as tf




def Train_data():
    """
    Creates dataset for training,validating,testing
    :return:
    train_data : training dataset
    validate_data : validation dataset
    test_data : test dataset
    """
    print ("loading train data ...")
    time_start = time.time()
    data_root = '/media/keziwen/86AA9651AA963E1D'
    with h5py.File(join(data_root, './data/train_real2.h5')) as f:
        data_real = f['train_real'][:]
    num, nt, ny, nx = data_real.shape
    data_real = np.transpose(data_real, (0, 1, 3, 2))
    with h5py.File(join(data_root, './data/train_imag2.h5')) as f:
        data_imag = f['train_imag'][:]
    num, nt, ny, nx = data_imag.shape
    data_imag = np.transpose(data_imag, (0, 1, 3, 2))
    data = data_real+1j*data_imag
    num_train = 15000
    num_validate = 2000
    train_data = data[0:num_train]
    validate_data = data[num_train:num_train+num_validate]

    train_data = np.random.permutation(train_data)

    time_end = time.time()
    print ('dataset has been created using {}s'.format(time_end-time_start))
    return train_data, validate_data

def Test_data():
    """
        Creates dataset for training,validating,testing
        :return:
        test_data : test dataset
        """
    print ("loading test data ...")
    time_start = time.time()
    data_root = '/media/keziwen/86AA9651AA963E1D'

    with h5py.File(join(data_root, './data/test_real2.h5')) as f:
        test_real = f['test_real'][:]
    with h5py.File(join(data_root, './data/test_imag2.h5')) as f:
        test_imag = f['test_imag'][:]
    test_real = np.transpose(test_real, (0, 1, 3, 2))
    test_imag = np.transpose(test_imag, (0, 1, 3, 2))
    test_data = test_real+1j*test_imag
    time_end = time.time()
    print ('dataset has been created using {}s'.format(time_end - time_start))
    return test_data

