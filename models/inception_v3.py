import tensorflow as tf


def build_inception_v3_model(print_model_summary=False):
    # TODO try using other model architectures, i.e., NASNet, EfficientNet etc.
    inception_v3_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet'
    )
    model_output = inception_v3_model.layers[-1].output

    image_features_model = tf.keras.Model(
        inception_v3_model.input, model_output
    )
    if print_model_summary:
        print(image_features_model.summary())

    return image_features_model
