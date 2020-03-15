import glob
import json
import os
import random
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf

from models import inception_v3

# set a seed for reproducibility
SEED = 2020

np.random.seed(SEED)
random.seed(SEED)

DATA_FOLDER = f"{os.path.abspath('.')}/data/"
ANNOTATIONS_FOLDER = f'{DATA_FOLDER}annotations/'
IMAGE_FOLDER = f'{DATA_FOLDER}train2014/'
FEATURES_FOLDER = f'{DATA_FOLDER}features/'


def download_mscoco_data():
    # download caption annotation files
    if not os.path.exists(ANNOTATIONS_FOLDER):
        annotation_zip = tf.keras.utils.get_file(
            fname='captions.zip',
            origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
            cache_subdir=DATA_FOLDER,
            extract=True
        )
        os.remove(annotation_zip)

    # download image files
    if not os.path.exists(IMAGE_FOLDER):
        image_zip = tf.keras.utils.get_file(
            fname='train2014.zip',
            origin='http://images.cocodataset.org/zips/train2014.zip',
            cache_subdir=DATA_FOLDER,
            extract=True
        )
        os.remove(image_zip)


def load_captions_and_image_names():
    # read the image annotations json file
    with open(f'{ANNOTATIONS_FOLDER}captions_train2014.json', 'r') as f:
        annotations = json.load(f)

    # store captions and image names in vectors
    all_captions = []
    all_image_names = []

    for annotation in annotations['annotations']:
        caption = f"<start> {annotation['caption']} <end>"
        image_id = annotation['image_id']
        full_coco_image_path = IMAGE_FOLDER + 'COCO_train2014_' + '%012d.jpg' % image_id

        all_image_names.append(full_coco_image_path)
        all_captions.append(caption)

    # shuffle the captions and image names together
    return shuffle(all_captions, all_image_names, random_state=SEED)


def pre_process_images():
    """
    Loads the MSCOCO images, uses an Inception V3 model to extract features
    using the ImageNet weights for each image and saves them as numpy arrays.

    There is a benefit of doing this as opposed to adding the Inception model
    to the tail of the main model that will be used later. Creating the
    features is a one-off cost that allows us to create a good model
    architecture that will perform the captioning using the computed features.
    We can than combine this optimised model architecture with a trainable
    Inception V3 model that computes features to further improve the overall
    model performance.
    """
    captions, image_names = load_captions_and_image_names()
    model = inception_v3.build_inception_v3_model()

    # get the unique images
    unique_image_names = sorted(set(image_names))

    image_dataset = tf.data.Dataset.from_tensor_slices(unique_image_names)
    image_dataset = image_dataset.map(
        compute_features_for_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    make_dirs_if_not_exists(FEATURES_FOLDER)

    for img, path in tqdm(image_dataset,
                          desc='Extracting features',
                          total=len(unique_image_names)):
        batch_features = model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1,
                                     batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            feature_path = p.numpy().decode('utf-8').replace(
                IMAGE_FOLDER, FEATURES_FOLDER).replace('jpg', 'npy')
            np.save(feature_path, bf.numpy())


def load_captioning_file_paths(top_k=5000):
    captions, image_names = load_captions_and_image_names()
    features_files = glob.glob1(FEATURES_FOLDER, '*.npy')
    features_files = [FEATURES_FOLDER + file for file in features_files]

    # choose the top words from the vocabulary
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=top_k,
        oov_token="<unk>",
        filters='!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ '
    )
    tokenizer.fit_on_texts(captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(captions)

    # pad each vector to the max_length of the captions
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
        train_seqs, padding='post'
    )

    # create training and validation sets using an 80-20 split
    features_file_train, features_file_val, captions_train, captions_val = train_test_split(
        features_files,
        cap_vector,
        test_size=0.2,
        random_state=SEED
    )

    return [features_file_train, features_file_val, captions_train, captions_val], tokenizer


def load_trained_tf_model(model_save_dir, model_name, use_best_val_loss=False):
    model_architecture_path = f'{model_save_dir}{model_name}.json'
    if use_best_val_loss:
        model_weights_path = f'{model_save_dir}{model_name}_best_val_loss.h5'
    else:
        model_weights_path = f'{model_save_dir}{model_name}.h5'

    print(f'Loading model architecture saved under: {model_architecture_path}')
    print(f'Loading model weighs saved under: {model_weights_path}')

    with open(model_architecture_path, 'r') as f:
        model = tf.python.keras.models.model_from_json(
            f.read(), custom_objects={'k': tf.keras.backend}
        )

    model.load_weights(model_weights_path)
    return model


def compute_features_for_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def make_dirs_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    download_mscoco_data()
