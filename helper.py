import numpy as np
import pickle
import scipy.stats as stats
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import tensorflow as tf
import random
from sklearn.preprocessing import LabelBinarizer
import pandas as pd


def normalize(x, args=[True, 1.0, True]):
    """
    Normalize an image data in the range of 0 to max value per channel
    : x: image data.  The image shape is (32, 32, 3)
    : return: normalized image data
    """
    # X' = a + (X - Xmin)(b-a)/(Xmax-Xmin)
    result = x.astype(float)

    # getting minimum and maximum values per channel [rrr,ggg,bbb]
    max_channel_values = result.max(axis=(0, 1)) * args[1]
    min_channel_values = result.min(axis=(0, 1)) if args[0] else [0, 0, 0]

    # iterating through each three channels (0,1,2)
    for i in range(result.shape[2]):
        channel = result[:, :, i]  # getting a channel as a slice

        bottom = min_channel_values[i]
        top = max_channel_values[i]

        channel_clip = np.clip(channel, bottom, top)

        min_max_diff = top - bottom

        if min_max_diff > 0:
            # '[::]' modifies a view of an array (in place)
            channel[::] = np.divide(np.subtract(channel_clip, bottom), min_max_diff)
        else:
            channel[::] = 1

    if args[2]:
        result = (result * 255).astype(np.uint8)

    return result


def normalize_by_confidence(x):
    """
    Normalize an image data in the range of 0 to max value per channel
    : x: image data.  The image shape is (32, 32, 3)
    : return: normalized image data
    """
    # X' = a + (X - Xmin)(b-a)/(Xmax-Xmin)
    result = x.astype(float)

    # getting minimum and maximum values per channel [rrr,ggg,bbb]
    # max_channel_values = result.max(axis=(0, 1))
    # min_channel_values = result.min(axis=(0, 1))

    # iterating through each three channels (0,1,2)
    for i in range(result.shape[2]):
        channel = result[:, :, i]  # getting a channel as a slice

        mean, sigma = np.mean(channel), np.std(channel)
        conf_int = stats.norm.interval(0.683, loc=mean, scale=sigma)

        bottom = conf_int[0]
        top = conf_int[1]

        channel_clip = np.clip(channel, bottom, top)

        min_max_diff = top - bottom

        if min_max_diff > 0:
            # '[::]' modifies a view of an array (in place)
            channel[::] = np.divide(np.subtract(channel_clip, bottom), min_max_diff)
        else:
            channel[::] = 1

    return result


def bw(x, heq=False):
    result = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

    result = result.reshape(result.shape[0], result.shape[1], 1)

    if heq:
        result = hist_eq(result)

    return result


def clahe(x, ch_to_heq=None):
    result = x.astype(np.uint8)

    clh = cv2.createCLAHE()

    if ch_to_heq is None:
        ch_to_heq = range(result.shape[2])

    for i in ch_to_heq:
        channel = result[:, :, i]
        channel[::] = clh.apply(channel[::])  # in place

    return result


def clahe_bw(x, heq=False):
    result = clahe(x)
    result = bw(result, heq)

    return result


def hist_eq(x, ch_to_heq=None):
    if ch_to_heq is None:
        ch_to_heq = range(x.shape[2])

    for i in ch_to_heq:
        x[:, :, i] = cv2.equalizeHist(x[:, :, i])

    return x


def change_colorspace(x, new_color_space, ch_to_heq=None):
    if ch_to_heq is None:
        return cv2.cvtColor(x, new_color_space)
    else:
        x = cv2.cvtColor(x, new_color_space)

        x = hist_eq(x, ch_to_heq)

        return x


def hls(x, ch_to_heq=None):
    return change_colorspace(x, cv2.COLOR_RGB2HLS, ch_to_heq)


def hsv(x, ch_to_heq=None):
    return change_colorspace(x, cv2.COLOR_RGB2HSV, ch_to_heq)


def yuv(x, ch_to_hec=None):
    return change_colorspace(x, cv2.COLOR_RGB2YUV, ch_to_hec)


def process_and_save(process, features, labels, filename, args=None):
    """
    process data and save it to file
    """

    results = []

    for i in range(len(features)):

        if args is not None:
            feature = process(features[i], args)
        else:
            feature = process(features[i])

        feature = norm(feature)

        results.append(feature)

    pickle.dump((results, labels), open(filename, 'wb'))


def norm(x):
    return np.divide(x, 255.0).astype(np.float32)


def augment_batch(src, label, mods_count=4, augment_args=[15, 3, 3, True, 4]):
    mods = []
    labels = []
    for _ in range(mods_count):
        angle = augment_args[0]
        shear = augment_args[1]
        trans = augment_args[2]
        augment_brigtness = augment_args[3]
        margin_delta = augment_args[4]
        mod = transform_image(img=src, ang_range=angle, shear_range=shear, trans_range=trans,
                              brightness=augment_brigtness, delta=margin_delta)
        mods.append(mod)
        labels.append(label)

    return mods, labels


def timing_stats_since(start, message='\n'):
    import time

    end = time.time()
    secs = end - start
    m = int(secs / 60)
    s = round((secs % 60), 3)

    print('{}Time elapsed: {} m {} s'.format(message, m, s))


# Idea by Vivek Yadav: https://github.com/vxy10/ImageAugmentation
def augment_brightness(image):
    hsv_img = hsv(image)

    random_brightness = np.random.uniform(0.75, 1.25)

    def clamp(a):
        return min(255, a * random_brightness)

    vfunc = np.vectorize(clamp)

    hsv_img[:, :, 2] = vfunc(hsv_img[:, :, 2])

    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)


# By Vivek Yadav: https://github.com/vxy10/ImageAugmentation
def transform_image(img, ang_range, shear_range, trans_range, brightness=False, delta=4):
    """
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    """

    w, h, d = img.shape

    nw = w + delta
    nh = h + delta

    img = cv2.resize(src=img, dsize=(nw, nh))

    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = img.shape
    rot_m = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    shear_m = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, rot_m, (cols, rows))
    img = cv2.warpAffine(img, trans_m, (cols, rows))
    img = cv2.warpAffine(img, shear_m, (cols, rows))

    # Brightness
    if brightness:
        img = augment_brightness(img)

    img = img[delta:nw, delta:nh, :]

    return img


def tensor_last_dim(x):
    tensor_shape = x.get_shape().as_list()
    return len(tensor_shape) - 1


def plot_samples(row_count, column_count, samples, labels, text_color='b', bg_color='c', randomize=True):
    plt.figure(figsize=(18, 18))
    gs1 = gs.GridSpec(row_count, row_count)
    gs1.update(wspace=0.01, hspace=0.01)

    for i in range(min(row_count * column_count, len(labels))):
        ax1 = plt.subplot(gs1[i])
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')

        if randomize:
            idx = np.random.randint(1, len(labels))
        else:
            idx = i

        plot_image(plt, samples[idx], [2, 4, text_color, bg_color, labels[idx]])

        plt.axis('off')
    plt.show()


def plot_image(pyplot, image, label_args=[2, 4, 'g', 'b', 'x']):
    pyplot.imshow(image)

    if label_args is not None:
        x = label_args[0]
        y = label_args[1]
        txt_color = label_args[2]
        bg_color = label_args[3]
        lbl_text = label_args[4]
        pyplot.text(x, y, str(lbl_text),
                    color=txt_color, backgroundcolor=bg_color)


def weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def augment(X, Y, augs_count=4, aug_args=[15, 3, 3, True, 4]):
    i = 0

    X_augmented = []
    Y_augmented = []

    samples_count = len(X)

    for x, y in zip(X, Y):

        X_augmented.append(x)
        Y_augmented.append(y)

        mod_samples, mod_labels = augment_batch(src=x, label=y, mods_count=augs_count, augment_args=aug_args)

        for mod, label in zip(mod_samples, mod_labels):
            X_augmented.append(mod)
            Y_augmented.append(label)

        i += 1

        if i % 2000 == 0:
            print('{} of {} items processed'.format(i, len(X)))

    return X_augmented, Y_augmented


def load_data(data_file):
    with open(data_file, mode='rb') as f:
        dataset = pickle.load(f)

    return dataset[0], dataset[1]


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_label_names(filename):
    """
    Load the label names from file
    """

    sig_names = pd.read_csv(filename)
    sig_labels = []

    # well it's ugly.
    for i in range(sig_names.shape[0]):
        sig_labels.append(sig_names.iloc[i, 1])

    return sig_labels, len(sig_labels)


def display_image_predictions(features,
                              labels,
                              predictions,
                              labels_legend_filename,
                              sample_count,
                              ton_n_predictions):
    label_names, n_classes = load_label_names(labels_legend_filename)

    fig, axes = plt.subplots(nrows=sample_count, ncols=2)
    fig.set_figheight(sample_count * 2)
    print('Softmax predictions:')

    margin = 0.05
    ind = np.arange(ton_n_predictions)
    width = 1. / ton_n_predictions

    for image_i, (feature, label_id, pred_indices, pred_values) in \
            enumerate(zip(features, labels, predictions.indices, predictions.values)):
        print(pred_values)
        pred_names = [label_names[pred_i] for pred_i in pred_indices]
        correct_name = label_names[label_id]

        axes[image_i][1].imshow(feature)
        axes[image_i][1].set_title(correct_name)
        axes[image_i][1].set_axis_off()

        axes[image_i][0].barh(ind + margin, pred_values[::-1], width)
        axes[image_i][0].set_yticks(ind + margin)
        axes[image_i][0].set_yticklabels(pred_names[::-1])
        axes[image_i][0].set_xticks([0, 0.5, 1.0])


def one_hot_encode(x, category_count):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    x_count = len(x)
    result = np.zeros((x_count, category_count))
    result[np.arange(x_count), x] = 1

    return result


def test_model(test_data_file,
               model_file,
               tf_names,
               batch_size,
               samples_to_test_count,
               top_n,
               labels_legend_file):

    x_test, y_test = load_data(data_file=test_data_file)

    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        loader = tf.train.import_meta_graph(model_file + '.meta')
        loader.restore(sess, model_file)

        loaded_x = loaded_graph.get_tensor_by_name('{}:0'.format(tf_names['samples']))
        loaded_y = loaded_graph.get_tensor_by_name('{}:0'.format(tf_names['labels']))
        loaded_keep_prob = loaded_graph.get_tensor_by_name('{}:0'.format(tf_names['keep_p']))
        loaded_logits = loaded_graph.get_tensor_by_name('{}:0'.format(tf_names['output']))
        loaded_acc = loaded_graph.get_tensor_by_name('{}:0'.format(tf_names['acc']))

        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0

        for feature_batch, label_batch in batch_features_labels(x_test, y_test, batch_size):
            test_batch_acc_total += sess.run(loaded_acc,
                                             feed_dict={loaded_x: feature_batch,
                                                        loaded_y: label_batch,
                                                        loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total / test_batch_count))

        random_test_features, random_test_labels = \
            tuple(zip(*random.sample(list(zip(x_test, y_test)), samples_to_test_count)))

        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n),
            feed_dict={loaded_x: random_test_features,
                       loaded_y: random_test_labels,
                       loaded_keep_prob: 1.0})

        display_image_predictions(features=random_test_features,
                                  labels=random_test_labels,
                                  predictions=random_test_predictions,
                                  labels_legend_filename=labels_legend_file,
                                  sample_count=samples_to_test_count,
                                  ton_n_predictions=top_n)


# def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1, plt_num=1):
#     # Here make sure to preprocess your image_input in a way your network expects
#     # with size, normalization, ect if needed
#     # image_input =
#     # Note: x should be the same name as your network's tensorflow data placeholder variable
#     # If you get an error tf_activation is not defined it maybe
#     # having trouble accessing the variable from inside a function
#     activation = tf_activation.eval(session=sess, feed_dict={x: image_input})
#     featuremaps = activation.shape[3]
#     plt.figure(plt_num, figsize=(15, 15))
#     for featuremap in range(featuremaps):
#         plt.subplot(6, 8, featuremap + 1)  # sets the number of feature maps to show on each row and column
#         plt.title('FeatureMap ' + str(featuremap))  # displays the feature map number
#         if activation_min != -1 & activation_max != -1:
#             plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
#                        vmax=activation_max, cmap="gray")
#         elif activation_max != -1:
#             plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
#         elif activation_min != -1:
#             plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
#         else:
#             plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", cmap="gray")


# def feature_map(checkpoint, layer_names, img, layer_idx):
#
#     saver = tf.train.Saver()
#
#     with tf.Session() as sess:
#         saver.restore(sess, checkpoint)
#
#         layers = []
#         for l_name in layer_names:
#             layers.append(sess.graph.get_tensor_by_name(l_name))
#
#         if layer_idx < len(layers):
#             outputFeatureMap(img, layers[layer_idx])






