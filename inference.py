import tensorflow as tf
from kernel_Init import ComplexInit

def complex_conv3d(input_op, name, kd, kh, kw, n_out=32, dd=1, dh=1, dw=1, regularizer=[], ifactivate=True):
    n_in = input_op.get_shape()[-1].value // 2

    with tf.name_scope(name) as scope:
        kernel_init = ComplexInit(kernel_size=[kd, kh, kw],
                                  input_dim=n_in,
                                  weight_dim=3,
                                  nb_filters=n_out,
                                  criterion='he')
        kernel = tf.get_variable(scope + 'weights',
                                 shape=[kd, kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=kernel_init)
        if regularizer is not None:
            # weight_loss = tf.multiply(tf.nn.l2_loss(var), regularizer, name='weight_loss')
            tf.add_to_collection('losses', regularizer(kernel))

        bias_init = tf.constant(0.0, dtype=tf.float32, shape=[n_out*2])
        biases = tf.get_variable(scope + 'biases', dtype=tf.float32, initializer=bias_init)

        kernel_real = kernel[:, :, :, :, :n_out]
        kernel_imag = kernel[:, :, :, :, n_out:]
        cat_kernel_real = tf.concat([kernel_real, -kernel_imag], axis=-2)
        cat_kernel_imag = tf.concat([kernel_imag, kernel_real], axis=-2)
        cat_kernel_complex = tf.concat([cat_kernel_real, cat_kernel_imag], axis=-1)

        conv = tf.nn.conv3d(input_op, cat_kernel_complex, strides=[1, dd, dh, dw, 1], padding='SAME')
        conv_bias = tf.nn.bias_add(conv, biases)
        if ifactivate:
            output = tf.nn.relu(conv_bias)
        else:
            output = conv_bias
        return output



def conv_op(input_op, name, kd, kh, kw, n_out, dd, dh, dw, regularizer, ifactivate):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",
                                 shape=[kd, kh, kw, n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.keras.initializers.he_normal())
        if regularizer is not None:
            # weight_loss = tf.multiply(tf.nn.l2_loss(var), regularizer, name='weight_loss')
            tf.add_to_collection('losses', regularizer(kernel))
        conv = tf.nn.conv3d(input_op, kernel, strides=[1, dd, dh, dw, 1], padding='SAME')
        bias_init_var = tf.constant(0.0, dtype=tf.float32, shape=[n_out])
        biases = tf.Variable(bias_init_var, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        if ifactivate is True:
            activation = tf.nn.relu(z, name=scope)
        else:
            activation = z
        return activation

def real2complex(input_op, inv=False):
    if inv == False:
        return tf.complex(input_op[:, :, :, :, 0], input_op[:, :, :, :, 1])
    else:
        input_real = tf.cast(tf.real(input_op), dtype=tf.float32)
        input_imag = tf.cast(tf.imag(input_op), dtype=tf.float32)
        return tf.stack([input_real, input_imag], axis=4)

def dc(input_dc, mask, kspace, case=1):
    if case==1:
        image = real2complex(input_dc)
        scaling = tf.sqrt(tf.cast(tf.multiply(tf.shape(image)[1], tf.shape(image)[2]), dtype=tf.complex64))
        image_fft = tf.fft2d(image, 'fft2')
        image_fft_dc = tf.multiply((1-mask), image_fft) + kspace
        image_dc = tf.ifft2d(image_fft_dc, 'ifft2')
        image = real2complex(image_dc, inv=True)
        return image
    if case==2:
        kspace_complex = real2complex(input_dc)
        kspace_dc = tf.multiply((1-mask), kspace_complex) + kspace
        return kspace_dc


# KIKI-Net
def inference(mask, kspace, regularizer):
    '''
    :param input_op: 2-channel undersampling kspace
    :param mask: complex-value 1-channel mask
    :param kspace: complex-value(1-channel) undersampling kspace
    :param regularizer: L2 norm for weights
    :return: reconstruction
    '''
    # KIKI-Net
def inference(mask, kspace, regularizer):
    '''
    :param input_op: 2-channel undersampling kspace
    :param mask: complex-value 1-channel mask
    :param kspace: complex-value(1-channel) undersampling kspace
    :param regularizer: L2 norm for weights
    :return: reconstruction
    '''
    # kspace-Net
    """
    input_op = real2complex(kspace, inv=True)
    temp = input_op
    for i in range(1):
        conv1_k = complex_conv3d(temp, name='conv' + str(i + 1) + '_k_1',
                          kd=3, kh=3, kw=3, n_out=32,
                          dd=1, dh=1, dw=1,
                          regularizer=regularizer, ifactivate=True)
        conv2_k = complex_conv3d(conv1_k, name='conv' + str(i + 1) + '_k_2',
                          kd=3, kh=3, kw=3, n_out=32,
                          dd=1, dh=1, dw=1,
                          regularizer=regularizer, ifactivate=True)
        conv3_k = complex_conv3d(conv2_k, name='conv' + str(i + 1) + '_k_3',
                          kd=3, kh=3, kw=3, n_out=32,
                          dd=1, dh=1, dw=1,
                          regularizer=regularizer, ifactivate=True)
        conv4_k = complex_conv3d(conv3_k, name='conv' + str(i + 1) + '_k_4',
                          kd=3, kh=3, kw=3, n_out=32,
                          dd=1, dh=1, dw=1,
                          regularizer=regularizer, ifactivate=True)
        conv5_k = complex_conv3d(conv4_k, name='conv' + str(i + 1) + '_k_5',
                          kd=3, kh=3, kw=3, n_out=1,
                          dd=1, dh=1, dw=1,
                          regularizer=regularizer, ifactivate=False)
        k_net_out = dc(conv5_k, mask=mask, kspace=kspace, case=2) #complex value
        k_recon = real2complex(k_net_out, inv=True)
        temp = k_recon
        image_k_net = tf.ifft2d(k_net_out, 'ifft2_k_net')
        input_Image_Net = real2complex(image_k_net, inv=True)
        if i == 0:
            block_k_1 = input_Image_Net
        if i == 1:
            block_k_2 = input_Image_Net
        if i == 2:
            block_k_3 = input_Image_Net
        if i == 3:
            block_k_4 = input_Image_Net
    """
    input_op = real2complex(kspace, inv=True)
    conv1_k = complex_conv3d(input_op, name='conv1_k',
                             kd=3, kh=3, kw=3, n_out=32,
                             dd=1, dh=1, dw=1,
                             regularizer=regularizer, ifactivate=True)
    conv2_k = complex_conv3d(conv1_k, name='conv2_k',
                             kd=3, kh=3, kw=3, n_out=32,
                             dd=1, dh=1, dw=1,
                             regularizer=regularizer, ifactivate=True)

    conv3_k = complex_conv3d(conv2_k, name='conv3_k',
                             kd=3, kh=3, kw=3, n_out=32,
                             dd=1, dh=1, dw=1,
                             regularizer=regularizer, ifactivate=True)

    conv4_k = complex_conv3d(conv3_k, name='conv4_k',
                             kd=3, kh=3, kw=3, n_out=32,
                             dd=1, dh=1, dw=1,
                             regularizer=regularizer, ifactivate=True)
    conv5_k = complex_conv3d(conv4_k, name='conv5_k',
                             kd=3, kh=3, kw=3, n_out=32,
                             dd=1, dh=1, dw=1,
                             regularizer=regularizer, ifactivate=True)
    conv6_k = complex_conv3d(conv5_k, name='conv6_k',
                             kd=3, kh=3, kw=3, n_out=32,
                             dd=1, dh=1, dw=1,
                             regularizer=regularizer, ifactivate=True)
    conv7_k = complex_conv3d(conv6_k, name='conv7_k',
                             kd=3, kh=3, kw=3, n_out=32,
                             dd=1, dh=1, dw=1,
                             regularizer=regularizer, ifactivate=True)
    conv8_k = complex_conv3d(conv7_k, name='conv8_k',
                             kd=3, kh=3, kw=3, n_out=32,
                             dd=1, dh=1, dw=1,
                             regularizer=regularizer, ifactivate=True)
    conv9_k = complex_conv3d(conv8_k, name='conv9_k',
                             kd=3, kh=3, kw=3, n_out=32,
                             dd=1, dh=1, dw=1,
                             regularizer=regularizer, ifactivate=True)
    conv10_k = complex_conv3d(conv9_k, name='conv10_k',
                             kd=3, kh=3, kw=3, n_out=1,
                             dd=1, dh=1, dw=1,
                             regularizer=regularizer, ifactivate=False)
    k_net_out = dc(conv10_k, mask=mask, kspace=kspace, case=2)  # complex value
    image_k_net = tf.ifft2d(k_net_out, 'ifft2_k_net')
    input_Image_Net = real2complex(image_k_net, inv=True)

    # D5C4
    temp = input_Image_Net
    for i in range(4):
        conv_1 = complex_conv3d(temp, name='conv' + str(i + 1) + '_1',
                         kd=3, kh=3, kw=3, n_out=32,
                         dd=1, dh=1, dw=1,
                         regularizer=regularizer, ifactivate=True)
        conv_2 = complex_conv3d(conv_1, name='conv' + str(i + 1) + '_2',
                         kd=3, kh=3, kw=3, n_out=32,
                         dd=1, dh=1, dw=1,
                         regularizer=regularizer, ifactivate=True)
        conv_3 = complex_conv3d(conv_2, name='conv' + str(i + 1) + '_3',
                         kd=3, kh=3, kw=3, n_out=32,
                         dd=1, dh=1, dw=1,
                         regularizer=regularizer, ifactivate=True)
        conv_4 = complex_conv3d(conv_3, name='conv' + str(i + 1) + '_4',
                         kd=3, kh=3, kw=3, n_out=32,
                         dd=1, dh=1, dw=1,
                         regularizer=regularizer, ifactivate=True)

        conv_5 = complex_conv3d(conv_4, name='conv' + str(i + 1) + '_5',
                                kd=3, kh=3, kw=3, n_out=32,
                                dd=1, dh=1, dw=1,
                                regularizer=regularizer, ifactivate=True)
        conv_6 = complex_conv3d(conv_5, name='conv' + str(i + 1) + '_6',
                                kd=3, kh=3, kw=3, n_out=32,
                                dd=1, dh=1, dw=1,
                                regularizer=regularizer, ifactivate=True)
        conv_7 = complex_conv3d(conv_6, name='conv' + str(i + 1) + '_7',
                                kd=3, kh=3, kw=3, n_out=32,
                                dd=1, dh=1, dw=1,
                                regularizer=regularizer, ifactivate=True)
        conv_8 = complex_conv3d(conv_7, name='conv' + str(i + 1) + '_8',
                                kd=3, kh=3, kw=3, n_out=32,
                                dd=1, dh=1, dw=1,
                                regularizer=regularizer, ifactivate=True)
        conv_9 = complex_conv3d(conv_8, name='conv' + str(i + 1) + '_9',
                                kd=3, kh=3, kw=3, n_out=32,
                                dd=1, dh=1, dw=1,
                                regularizer=regularizer, ifactivate=True)
        conv_10 = complex_conv3d(conv_9, name='conv' + str(i + 1) + '_10',
                         kd=3, kh=3, kw=3, n_out=1,
                         dd=1, dh=1, dw=1,
                         regularizer=regularizer, ifactivate=False)

        block = temp + conv_10
        block_dc = dc(block, mask=mask, kspace=kspace, case=1)
        temp = block_dc
        if i == 0:
            block_1 = temp
        if i == 1:
            block_2 = temp
        if i == 2:
            block_3 = temp


    return temp, block_1, block_2, block_3, input_Image_Net






