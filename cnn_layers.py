import tensorflow as tf
import helper as aux


def conv2d_layer(x_tensor, num_outputs,
                 conv_args=[[3, 3], [1, 1], 'VALID'],
                 pool_args=None, keep_prob=1.0, l2norm=False,
                 activation='relu', layer_name=None):
    conv_ksize = conv_args[0]
    conv_strides = conv_args[1]

    weight = aux.weights([conv_ksize[0], conv_ksize[1], x_tensor.get_shape()[3].value, num_outputs])
    bias = aux.biases(num_outputs)

    # Ugly but allows not to fix all models without explicit layer names as tf.nn.conv2d most likely
    # won't accept None value as an argument.
    # TODO: Fix this!
    if layer_name is not None:
        conv_layer = tf.nn.conv2d(x_tensor, weight,
                                  strides=[1, conv_strides[0], conv_strides[1], 1],
                                  padding=conv_args[2], name=layer_name)
    else:
        conv_layer = tf.nn.conv2d(x_tensor, weight,
                                  strides=[1, conv_strides[0], conv_strides[1], 1],
                                  padding=conv_args[2])

    conv_layer = tf.nn.bias_add(conv_layer, bias)

    if activation == 'relu':
        conv_layer = tf.nn.relu(conv_layer)
    elif activation == 'elu':
        conv_layer = tf.nn.elu(conv_layer)

    if pool_args is not None:
        pool_ksize = pool_args[0]
        pool_strides = pool_args[1]

        conv_layer = tf.nn.max_pool(conv_layer,
                                    ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                                    strides=[1, pool_strides[0], pool_strides[1], 1],
                                    padding=pool_args[2])

    if l2norm:
        tldim = aux.tensor_last_dim(conv_layer)
        conv_layer = tf.nn.l2_normalize(x=conv_layer, dim=tldim, epsilon=1e-12)

    conv_layer = tf.nn.dropout(conv_layer, keep_prob)

    return conv_layer, weight


def flat_layer(x_tensor):
    shape = x_tensor.get_shape().as_list()
    flattened = shape[1] * shape[2] * shape[3]
    result = tf.reshape(x_tensor, [-1, flattened])

    return result


def fully_connected_layer(x_tensor, num_outputs, keep_prob=1.0,
                          output=False, l2norm=True,
                          activation='relu', layer_name=None):
    tensor_shape = x_tensor.get_shape().as_list()

    # One can't be sure on dimensionality of the passed in tensor.
    # Thus iterating through all possible dimensions skipping the primary batch_size dimension.
    weight_dim = 1
    for i in range(len(tensor_shape) - 1):
        weight_dim *= tensor_shape[i + 1]

    weight = aux.weights([weight_dim, num_outputs])
    bias = aux.biases(num_outputs)

    if layer_name is not None:
        fc = tf.add(tf.matmul(x_tensor, weight), bias, name=layer_name)
    else:
        fc = tf.add(tf.matmul(x_tensor, weight), bias)

    if not output:
        if activation == 'relu':
            fc = tf.nn.relu(fc)
        elif activation == 'elu':
            fc = tf.nn.elu(fc)

    if l2norm:
        tldim = aux.tensor_last_dim(fc)
        fc = tf.nn.l2_normalize(x=fc, dim=tldim, epsilon=1e-12)

    fc = tf.nn.dropout(fc, keep_prob)

    return fc, weight
