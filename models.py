import cnn_layers as cnnl
import tensorflow as tf


def mlp(x):
    flat = cnnl.flat_layer(x_tensor=x)
    fc1, w1 = cnnl.fully_connected_layer(x_tensor=flat, num_outputs=6144)
    fc2, w2 = cnnl.fully_connected_layer(x_tensor=fc1, num_outputs=4096)
    out, wo = cnnl.fully_connected_layer(x_tensor=fc2, num_outputs=43, output=True)

    weights = [w1, w2, wo]

    return out, weights


# some ad-hoc
def cnn(x, keep_probability):
    pool_k_size = [2, 2]
    pool_strides = [2, 2]
    p_args = [pool_k_size, pool_strides, 'SAME']

    k1_size = [1, 1]
    k1_strides = [1, 1]
    k1_count = 3
    k1_args = [k1_size, k1_strides, 'SAME']

    k2_size = [5, 5]
    k2_strides = [1, 1]
    k2_count = 96  # Low-level patterns
    k2_args = [k2_size, k2_strides, 'SAME']

    k3_size = [5, 5]
    k3_strides = [1, 1]
    k3_count = 128  # More complex ideas
    k3_args = [k3_size, k3_strides, 'SAME']

    k4_size = [5, 5]
    k4_strides = [1, 1]
    k4_count = 256  # Shapes
    k4_args = [k4_size, k4_strides, 'SAME']

    fc1_count = 1024  # Empiricaly seems to be reasonable

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=k1_count,
                                  conv_args=k1_args)

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1,
                                  num_outputs=k2_count,
                                  conv_args=k2_args,
                                  keep_prob=keep_probability)

    cv2_pool = tf.nn.max_pool(cv2,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])


    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2_pool,
                                  num_outputs=k3_count,
                                  conv_args=k3_args,
                                  keep_prob=keep_probability)

    cv3_pool = tf.nn.max_pool(cv3,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    cv4, cvw4 = cnnl.conv2d_layer(x_tensor=cv3_pool,
                                  num_outputs=k4_count,
                                  conv_args=k4_args,
                                  keep_prob=keep_probability)

    cv4_pool = tf.nn.max_pool(cv4,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    flat_cv = cnnl.flat_layer(x_tensor=cv4_pool)

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat_cv,
                                           num_outputs=fc1_count,
                                           keep_prob=keep_probability)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc1,
                                            num_outputs=43,
                                            output=True)

    weights = [cvw1, cvw2, cvw3, cvw4, fcw1, ow]
    return output, weights, 'cnn'


#forget it. Really.
def eccv(x, keep_probability):
    pool_k_size = [2, 2]
    pool_strides = [2, 2]
    p_args = [pool_k_size, pool_strides, 'SAME']

    k1_size = [1, 1]  # basically split into 3 color spaces
    k1_strides = [1, 1]
    k1_count = 3
    k1_args = [k1_size, k1_strides, 'SAME']

    k2_size = [5, 5]
    k2_strides = [1, 1]
    k2_count = 96
    k2_args = [k2_size, k2_strides, 'SAME']

    k3_size = [3, 3]
    k3_strides = [1, 1]
    k3_count = 256
    k3_args = [k3_size, k3_strides, 'SAME']

    k4_size = [3, 3]
    k4_strides = [1, 1]
    k4_count = 384
    k4_args = [k4_size, k4_strides, 'SAME']

    k5_size = [3, 3]
    k5_strides = [1, 1]
    k5_count = 256
    k5_args = [k5_size, k5_strides, 'SAME']

    fc1_count = 2048  # Empiricaly seems to be reasonable
    fc2_count = 1024

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=k1_count,
                                  conv_args=k1_args,
                                  keep_prob=keep_probability)

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1,
                                  num_outputs=k2_count,
                                  conv_args=k2_args,
                                  keep_prob=keep_probability)
    cv2_pool = tf.nn.max_pool(cv2,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2_pool,
                                  num_outputs=k3_count,
                                  conv_args=k3_args,
                                  keep_prob=keep_probability)

    cv4, cvw4 = cnnl.conv2d_layer(x_tensor=cv3,
                                  num_outputs=k4_count,
                                  conv_args=k4_args,
                                  keep_prob=keep_probability)

    cv5, cvw5 = cnnl.conv2d_layer(x_tensor=cv4,
                                  num_outputs=k5_count,
                                  conv_args=k5_args,
                                  keep_prob=keep_probability)
    cv5_pool = tf.nn.max_pool(cv5,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])


    flat_cv = cnnl.flat_layer(x_tensor=cv5_pool)

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat_cv,
                                           num_outputs=fc1_count,
                                           keep_prob=keep_probability)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=fc2_count,
                                           keep_prob=keep_probability)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    weights = [cvw1, cvw2, cvw3, cvw4, cvw5, fcw1, fcw2, ow]
    return output, weights, 'eccv'


def cnnt_lenet(x, keep_probability):
    kernel_size = [5, 5]
    kernel_strides = [1, 1]
    pool_k_size = [2, 2]
    pool_strides = [2, 2]
    k_args = [kernel_size, kernel_strides, 'VALID']
    p_args = [pool_k_size, pool_strides, 'VALID']

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=6,
                                  conv_args=k_args,
                                  keep_prob=keep_probability)
    cv1_pool = tf.nn.max_pool(cv1,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1_pool,
                                  num_outputs=16,
                                  conv_args=k_args,
                                  keep_prob=keep_probability)
    cv2_pool = tf.nn.max_pool(cv2,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    flat1 = cnnl.flat_layer(x_tensor=cv2_pool)

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=120,
                                           keep_prob=keep_probability)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=84,
                                           keep_prob=keep_probability)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    weights = [cvw1, cvw2, fcw1, fcw2, ow]

    return output, weights, 'lenet'


# By Pierre Sermanet and Yann LeCun
# with fancy YUV distribution prior to convolution. At least as I got it.
def sermanet(x, keep_probability):
    """
    cv1: Tensor("dropout/mul:0", shape=(?, 16, 16, 108), dtype=float32)
    cv2: Tensor("dropout_1/mul:0", shape=(?, 8, 8, 108), dtype=float32)
    cv1_branch: Tensor("MaxPool_2:0", shape=(?, 8, 8, 108), dtype=float32)
    fc1: Tensor("dropout_2/mul:0", shape=(?, 100), dtype=float32)
    fc2: Tensor("dropout_3/mul:0", shape=(?, 50), dtype=float32)
    output: Tensor("l2_normalize_2:0", shape=(?, 43), dtype=float32)

    :param x:
    :param keep_probability:
    :return:
    """

    k_size = [4, 4]
    k_strides = [1, 1]
    k_args = [k_size, k_strides, 'SAME']

    pool_k_size = [2, 2]
    pool_strides = [2, 2]
    p_args = [pool_k_size, pool_strides, 'SAME']

    y_ch_features = 100
    uv_ch_features = 8

    cv2_features = 108

    fc1_count = 100
    fc2_count = 50

    tf_ver = 1 if tf.__version__.startswith('1.0') else 0

    # 1st stage

    y, wy = cnnl.conv2d_layer(x_tensor=tf.expand_dims(x[:, :, :, 0], axis=3),
                              num_outputs=y_ch_features,
                              conv_args=k_args)

    uv, wuv = cnnl.conv2d_layer(x_tensor=x[:, :, :, 1:],
                                num_outputs=uv_ch_features,
                                conv_args=k_args)

    cv1 = tf.concat([y, uv], 3) if tf_ver else tf.concat(3, [y, uv])

    cv1_pool = tf.nn.max_pool(cv1,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    # 2nd stage
    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1_pool,
                                  num_outputs=cv2_features,
                                  conv_args=k_args,
                                  keep_prob=keep_probability)

    cv2_pool = tf.nn.max_pool(cv2,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    cv1_branch = tf.nn.max_pool(cv1_pool,
                                ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                                strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    flat_cv2 = cnnl.flat_layer(x_tensor=cv2_pool)
    flat_cv1b = cnnl.flat_layer(x_tensor=cv1_branch)


    flat1 = tf.concat([flat_cv2, flat_cv1b], 1) if tf_ver == 1 else tf.concat(1, [flat_cv2, flat_cv1b])

    # Classifier
    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=fc1_count,
                                           keep_prob=keep_probability)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=fc2_count,
                                           keep_prob=keep_probability)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    weights = [wy, wuv, cvw2, fcw1, fcw2, ow]

    print('cv1: {}'.format(cv1_pool))
    print('cv2: {}'.format(cv2_pool))
    print('cv1_branch: {}'.format(cv1_branch))
    print('fc1: {}'.format(fc1))
    print('fc2: {}'.format(fc2))
    print('output: {}'.format(output))

    return output, weights, 'sermanet'


# Fancy convolutional merging to fully-connected
def fancy(x, keep_probability):
    pool_k_size = [2, 2]
    pool_strides = [2, 2]

    p_args = [pool_k_size, pool_strides, 'SAME']

    k1_size = [1, 1]
    k1_strides = [1, 1]
    k1_count = 3
    k1_args = [k1_size, k1_strides, 'SAME']

    k2_size = [5, 5]
    k2_strides = [1, 1]
    k2_count = 32
    k2_args = [k2_size, k2_strides, 'SAME']

    k3_size = [5, 5]
    k3_strides = [1, 1]
    k3_count = 32
    k3_args = [k3_size, k3_strides, 'SAME']

    k4_size = [5, 5]
    k4_strides = [1, 1]
    k4_count = 64
    k4_args = [k4_size, k4_strides, 'SAME']

    k5_size = [5, 5]
    k5_strides = [1, 1]
    k5_count = 64
    k5_args = [k5_size, k5_strides, 'SAME']

    k6_size = [5, 5]
    k6_strides = [1, 1]
    k6_count = 128
    k6_args = [k6_size, k6_strides, 'SAME']

    k7_size = [5, 5]
    k7_strides = [1, 1]
    k7_count = 128
    k7_args = [k7_size, k7_strides, 'SAME']

    fc1_count = 1024
    fc2_count = 1024

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=k1_count,
                                  conv_args=k1_args)

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1,
                                  num_outputs=k2_count,
                                  conv_args=k2_args)

    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2,
                                  num_outputs=k3_count,
                                  conv_args=k3_args,
                                  keep_prob=keep_probability)
    cv3_pool = tf.nn.max_pool(cv3,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    cv4, cvw4 = cnnl.conv2d_layer(x_tensor=cv3_pool,
                                  num_outputs=k4_count,
                                  conv_args=k4_args)

    cv5, cvw5 = cnnl.conv2d_layer(x_tensor=cv4,
                                  num_outputs=k5_count,
                                  conv_args=k5_args,
                                  keep_prob=keep_probability)
    cv5_pool = tf.nn.max_pool(cv5,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    cv6, cvw6 = cnnl.conv2d_layer(x_tensor=cv5_pool,
                                  num_outputs=k6_count,
                                  conv_args=k6_args)

    cv7, cvw7 = cnnl.conv2d_layer(x_tensor=cv6,
                                  num_outputs=k7_count,
                                  conv_args=k7_args,
                                  keep_prob=keep_probability)
    cv7_pool = tf.nn.max_pool(cv7,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    flat_cv3 = cnnl.flat_layer(x_tensor=cv3_pool)
    flat_cv5 = cnnl.flat_layer(x_tensor=cv5_pool)
    flat_cv7 = cnnl.flat_layer(x_tensor=cv7_pool)

    if tf.__version__.startswith('1.0'):
        flat1 = tf.concat([flat_cv3, flat_cv5, flat_cv7], 1)
    else:
        flat1 = tf.concat(1, [flat_cv3, flat_cv5, flat_cv7])

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=fc1_count,
                                           keep_prob=keep_probability)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=fc2_count,
                                           keep_prob=keep_probability)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    weights = [cvw1, cvw2, cvw3, cvw4, cvw5, cvw6, cvw7, fcw1, fcw2, ow]

    return output, weights, 'fancy_v1'


# Fancy convolutional merging to fully-connected
def fancy_v2(x, keep_probability):
    pool_k_size = [2, 2]
    pool_strides = [2, 2]

    p_args = [pool_k_size, pool_strides, 'SAME']

    k1_size = [1, 1]
    k1_strides = [1, 1]
    k1_count = 3
    k1_args = [k1_size, k1_strides, 'SAME']

    k2_size = [5, 5]
    k2_strides = [1, 1]
    k2_count = 48
    k2_args = [k2_size, k2_strides, 'SAME']

    k3_size = [5, 5]
    k3_strides = [1, 1]
    k3_count = 48
    k3_args = [k3_size, k3_strides, 'SAME']

    k4_size = [5, 5]
    k4_strides = [1, 1]
    k4_count = 128
    k4_args = [k4_size, k4_strides, 'SAME']

    k5_size = [5, 5]
    k5_strides = [1, 1]
    k5_count = 128
    k5_args = [k5_size, k5_strides, 'SAME']

    fc1_count = 1024
    fc2_count = 1024

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=k1_count,
                                  conv_args=k1_args)

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1,
                                  num_outputs=k2_count,
                                  conv_args=k2_args)

    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2,
                                  num_outputs=k3_count,
                                  conv_args=k3_args,
                                  keep_prob=keep_probability)
    cv3_pool = tf.nn.max_pool(cv3,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    cv4, cvw4 = cnnl.conv2d_layer(x_tensor=cv3_pool,
                                  num_outputs=k4_count,
                                  conv_args=k4_args)

    cv5, cvw5 = cnnl.conv2d_layer(x_tensor=cv4,
                                  num_outputs=k5_count,
                                  conv_args=k5_args,
                                  keep_prob=keep_probability)
    cv5_pool = tf.nn.max_pool(cv5,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    flat_cv3 = cnnl.flat_layer(x_tensor=cv3_pool)
    flat_cv5 = cnnl.flat_layer(x_tensor=cv5_pool)

    if tf.__version__.startswith('1.0'):
        flat1 = tf.concat([flat_cv3, flat_cv5], 1)
    else:
        flat1 = tf.concat(1, [flat_cv3, flat_cv5])

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=fc1_count,
                                           keep_prob=keep_probability)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=fc2_count,
                                           keep_prob=keep_probability)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    weights = [cvw1, cvw2, cvw3, cvw4, cvw5, fcw1, fcw2, ow]

    print(cv1)
    print(cv2)
    print(cv3)
    print(cv4)
    print(cv5)
    print(fc1)
    print(fc2)
    print(output)

    return output, weights, 'fancy_v2'


# 3x3 cv2, cv3 kernel, valid padding, no dropout
def traffic_net_v2(x, keep_probability, activation='relu'):
    kernel_size = [5, 5]
    kernel_strides = [1, 1]
    pool_k_size = [2, 2]
    pool_strides = [2, 2]
    k_args = [kernel_size, kernel_strides, 'SAME']
    p_args = [pool_k_size, pool_strides, 'SAME']

    k2_args = [[3, 3], [1, 1], 'VALID']
    k3_args = [[3, 3], [1, 1], 'VALID']

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=64,
                                  conv_args=k_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability,
                                  activation=activation)

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1,
                                  num_outputs=128,
                                  conv_args=k2_args,
                                  pool_args=p_args,
                                  activation=activation)

    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2,
                                  num_outputs=192,
                                  conv_args=k3_args,
                                  pool_args=p_args,
                                  activation=activation)

    flat1 = cnnl.flat_layer(x_tensor=cv3)

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=2048,
                                           keep_prob=keep_probability,
                                           activation=activation)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=1024,
                                           keep_prob=keep_probability,
                                           activation=activation)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    print(cv1)
    print(cv2)
    print(cv3)
    print(fc1)
    print(fc2)
    print(output)

    weights = [cvw1, cvw2, cvw3, fcw1, fcw2, ow]

    return output, weights, 'traffic_net_v2_{}'.format(activation)


# 3x3 cv2, cv3 kernel, valid padding, no dropout
def traffic_net_v2_full_dropout(x, keep_probability, activation='relu'):
    kernel_size = [5, 5]
    kernel_strides = [1, 1]
    pool_k_size = [2, 2]
    pool_strides = [2, 2]
    k_args = [kernel_size, kernel_strides, 'SAME']
    p_args = [pool_k_size, pool_strides, 'SAME']

    k2_args = [[3, 3], [1, 1], 'VALID']
    k3_args = [[3, 3], [1, 1], 'VALID']

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=64,
                                  conv_args=k_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability,
                                  activation=activation)

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1,
                                  num_outputs=128,
                                  conv_args=k2_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability,
                                  activation=activation)

    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2,
                                  num_outputs=192,
                                  conv_args=k3_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability,
                                  activation=activation)

    flat1 = cnnl.flat_layer(x_tensor=cv3)

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=2048,
                                           keep_prob=keep_probability,
                                           activation=activation)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=1024,
                                           keep_prob=keep_probability,
                                           activation=activation)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    print(x)
    print(cv1)
    print(cv2)
    print(cv3)
    print(fc1)
    print(fc2)
    print(output)

    weights = [cvw1, cvw2, cvw3, fcw1, fcw2, ow]

    return output, weights, 'traffic_net_v2_full_dropout{}'.format(activation)


# 3x3 cv2, cv3 kernel, same padding, no dropout
def traffic_net_v2_full_dropout_same(x, keep_probability, activation='relu'):
    kernel_size = [5, 5]
    kernel_strides = [1, 1]
    pool_k_size = [2, 2]
    pool_strides = [2, 2]
    k_args = [kernel_size, kernel_strides, 'SAME']
    p_args = [pool_k_size, pool_strides, 'SAME']

    k2_args = [[3, 3], [1, 1], 'SAME']
    k3_args = [[3, 3], [1, 1], 'SAME']

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=64,
                                  conv_args=k_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability,
                                  activation=activation)

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1,
                                  num_outputs=128,
                                  conv_args=k2_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability,
                                  activation=activation)

    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2,
                                  num_outputs=192,
                                  conv_args=k3_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability,
                                  activation=activation)

    flat1 = cnnl.flat_layer(x_tensor=cv3)

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=2048,
                                           keep_prob=keep_probability,
                                           activation=activation)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=1024,
                                           keep_prob=keep_probability,
                                           activation=activation)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    print(x)
    print(cv1)
    print(cv2)
    print(cv3)
    print(fc1)
    print(fc2)
    print(output)

    weights = [cvw1, cvw2, cvw3, fcw1, fcw2, ow]

    return output, weights, 'traffic_net_v2_full_dropout{}'.format(activation)


# 3x3 cv2, cv3 kernel, valid padding, no dropout
def traffic_net_v2_pool_split(x, keep_probability):
    kernel_size = [5, 5]
    kernel_strides = [1, 1]
    pool_k_size = [2, 2]
    pool_strides = [2, 2]
    k_args = [kernel_size, kernel_strides, 'SAME']
    p_args = [pool_k_size, pool_strides, 'SAME']

    k2_args = [[3, 3], [1, 1], 'VALID']
    k3_args = [[3, 3], [1, 1], 'VALID']

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=64,
                                  conv_args=k_args,
                                  keep_prob=keep_probability)
    cv1_pool = tf.nn.max_pool(cv1,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1_pool,
                                  num_outputs=128,
                                  conv_args=k2_args)
    cv2_pool = tf.nn.max_pool(cv2,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2_pool,
                                  num_outputs=192,
                                  conv_args=k3_args)
    cv3_pool = tf.nn.max_pool(cv3,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    flat1 = cnnl.flat_layer(x_tensor=cv3_pool)

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=2048,
                                           keep_prob=keep_probability)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=1024,
                                           keep_prob=keep_probability)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    print(cv1)
    print(cv2)
    print(cv3)
    print(fc1)
    print(fc2)
    print(output)

    weights = [cvw1, cvw2, cvw3, fcw1, fcw2, ow]

    return output, weights, 'traffic_net_v2_pool_split'

def traffic_net_v2_pool_split(x, keep_probability):
    kernel_size = [5, 5]
    kernel_strides = [1, 1]
    pool_k_size = [2, 2]
    pool_strides = [2, 2]
    k_args = [kernel_size, kernel_strides, 'SAME']
    p_args = [pool_k_size, pool_strides, 'SAME']

    k2_args = [[3, 3], [1, 1], 'VALID']
    k3_args = [[3, 3], [1, 1], 'VALID']

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=64,
                                  conv_args=k_args,
                                  keep_prob=keep_probability)
    cv1_pool = tf.nn.max_pool(cv1,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1_pool,
                                  num_outputs=128,
                                  conv_args=k2_args)
    cv2_pool = tf.nn.max_pool(cv2,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2_pool,
                                  num_outputs=192,
                                  conv_args=k3_args)
    cv3_pool = tf.nn.max_pool(cv3,
                              ksize=[1, pool_k_size[0], pool_k_size[1], 1],
                              strides=[1, pool_strides[0], pool_strides[1], 1], padding=p_args[2])

    flat1 = cnnl.flat_layer(x_tensor=cv3_pool)

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=2048,
                                           keep_prob=keep_probability)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=1024,
                                           keep_prob=keep_probability)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    print(cv1)
    print(cv2)
    print(cv3)
    print(fc1)
    print(fc2)
    print(output)

    weights = [cvw1, cvw2, cvw3, fcw1, fcw2, ow]

    return output, weights, 'traffic_net_v2_pool_split'

"""

def traffic_net(x, keep_probability):
    kernel_size = [5, 5]
    kernel_strides = [1, 1]
    pool_k_size = [2, 2]
    pool_strides = [2, 2]
    k_args = [kernel_size, kernel_strides, 'SAME']
    p_args = [pool_k_size, pool_strides, 'SAME']

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=64,
                                  conv_args=k_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability)

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1,
                                  num_outputs=128,
                                  conv_args=k_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability)

    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2,
                                  num_outputs=192,
                                  conv_args=k_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability)

    flat1 = cnnl.flat_layer(x_tensor=cv3)

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=2048,
                                           keep_prob=keep_probability)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=1024,
                                           keep_prob=keep_probability)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    weights = [cvw1, cvw2, cvw3, fcw1, fcw2, ow]

    return output, weights


# 3x3 cv3 kernel, valid padding, no dropout
def traffic_net_v1(x, keep_probability):
    kernel_size = [5, 5]
    kernel_strides = [1, 1]
    pool_k_size = [2, 2]
    pool_strides = [2, 2]
    k_args = [kernel_size, kernel_strides, 'SAME']
    p_args = [pool_k_size, pool_strides, 'SAME']

    k3_args = [[3, 3], [1, 1], 'VALID']

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=64,
                                  conv_args=k_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability)

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1,
                                  num_outputs=128,
                                  conv_args=k_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability)

    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2,
                                  num_outputs=192,
                                  conv_args=k3_args,
                                  pool_args=p_args)

    flat1 = cnnl.flat_layer(x_tensor=cv3)

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=2048,
                                           keep_prob=keep_probability)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=1024,
                                           keep_prob=keep_probability)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    print(cv1)
    print(cv2)
    print(cv3)
    print(fc1)
    print(fc2)
    print(output)

    weights = [cvw1, cvw2, cvw3, fcw1, fcw2, ow]

    return output, weights, 'traffic_net_v1'


# 3x3 cv2, cv3 kernel, valid padding, no dropout
def traffic_net_v2(x, keep_probability):
    kernel_size = [5, 5]
    kernel_strides = [1, 1]
    pool_k_size = [2, 2]
    pool_strides = [2, 2]
    k_args = [kernel_size, kernel_strides, 'SAME']
    p_args = [pool_k_size, pool_strides, 'SAME']

    k2_args = [[3, 3], [1, 1], 'VALID']
    k3_args = [[3, 3], [1, 1], 'VALID']

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=64,
                                  conv_args=k_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability)

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1,
                                  num_outputs=128,
                                  conv_args=k2_args,
                                  pool_args=p_args)

    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2,
                                  num_outputs=192,
                                  conv_args=k3_args,
                                  pool_args=p_args)

    flat1 = cnnl.flat_layer(x_tensor=cv3)

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=2048,
                                           keep_prob=keep_probability)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=1024,
                                           keep_prob=keep_probability)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    print(cv1)
    print(cv2)
    print(cv3)
    print(fc1)
    print(fc2)
    print(output)

    weights = [cvw1, cvw2, cvw3, fcw1, fcw2, ow]

    return output, weights, 'traffic_net_v2'


# 3x3 cv2, cv3 kernel, same padding, no pooling
# pooling seems to be of no use on such a small images as it eats up dimentions almost completely
def traffic_net_v4(x, keep_probability):
    kernel_size = [5, 5]
    kernel_strides = [1, 1]
    pool_k_size = [2, 2]
    pool_strides = [2, 2]

    k_args = [kernel_size, kernel_strides, 'SAME']
    p_args = [pool_k_size, pool_strides, 'SAME']

    k2_args = [[3, 3], [1, 1], 'VALID']
    k3_args = [[3, 3], [1, 1], 'VALID']

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=80,
                                  conv_args=k_args,
                                  keep_prob=keep_probability)

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1,
                                  num_outputs=160,
                                  conv_args=k2_args,
                                  keep_prob=keep_probability)

    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2,
                                  num_outputs=192,
                                  conv_args=k3_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability)

    flat1 = cnnl.flat_layer(x_tensor=cv3)

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=2048,
                                           keep_prob=keep_probability)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=1024,
                                           keep_prob=keep_probability)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    print(cv1)
    print(cv2)
    print(cv3)
    print(fc1)
    print(fc2)
    print(output)

    weights = [cvw1, cvw2, cvw3, fcw1, fcw2, ow]

    return output, weights, 'traffic_net_v4'


def traffic_net_v5(x, keep_probability):
    kernel_size = [5, 5]
    kernel_strides = [1, 1]
    pool_k_size = [2, 2]
    pool_strides = [2, 2]

    k_args = [kernel_size, kernel_strides, 'SAME']
    p_args = [pool_k_size, pool_strides, 'SAME']

    k2_args = [[3, 3], [1, 1], 'VALID']
    k3_args = [[3, 3], [1, 1], 'VALID']

    cv1, cvw1 = cnnl.conv2d_layer(x_tensor=x,
                                  num_outputs=64,
                                  conv_args=k_args,
                                  keep_prob=keep_probability)

    cv2, cvw2 = cnnl.conv2d_layer(x_tensor=cv1,
                                  num_outputs=128,
                                  conv_args=k2_args,
                                  keep_prob=keep_probability)

    cv3, cvw3 = cnnl.conv2d_layer(x_tensor=cv2,
                                  num_outputs=192,
                                  conv_args=k3_args,
                                  pool_args=p_args,
                                  keep_prob=keep_probability)

    flat1 = cnnl.flat_layer(x_tensor=cv3)

    fc1, fcw1 = cnnl.fully_connected_layer(x_tensor=flat1,
                                           num_outputs=2048,
                                           keep_prob=keep_probability)

    fc2, fcw2 = cnnl.fully_connected_layer(x_tensor=fc1,
                                           num_outputs=1024,
                                           keep_prob=keep_probability)

    output, ow = cnnl.fully_connected_layer(x_tensor=fc2,
                                            num_outputs=43,
                                            output=True)

    print(cv1)
    print(cv2)
    print(cv3)
    print(fc1)
    print(fc2)
    print(output)

    weights = [cvw1, cvw2, cvw3, fcw1, fcw2, ow]

    return output, weights, 'traffic_net_v5'

"""


