import tensorflow as tf
import numpy as np

from utils import load_captioning_file_paths


def load_captioning_training_data(batch_size=64, buffer_size=1000, top_k=5000):
    [features_file_train, features_file_val, captions_train, captions_val], tokenizer = \
        load_captioning_file_paths(top_k)

    data_set = tf.data.Dataset.from_tensor_slices(
        (features_file_train, captions_train))

    # Use map to load the numpy files in parallel
    data_set = data_set.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    data_set = data_set.shuffle(buffer_size).batch(batch_size)
    data_set = data_set.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return data_set


def map_func(img_name, cap):
    """Loads the pre-processed Numpy arrays"""
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap
