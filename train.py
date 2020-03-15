import os
import time

from utils import download_mscoco_data, pre_process_images
from lib.data_loaders import load_captioning_training_data
from models import captioning_model

import tensorflow as tf

# Make sure we are using TensorFlow 2. If building from the Dockerfile, this
# won't be an issue.
assert tf.__version__.startswith('2.')

# Set the XLA compilation flags. This speeds up model training by compiling
# the model using a JIT compiler: https://www.tensorflow.org/xla
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

HYPER_PARAMETERS = {
    'batch_size': 64,
    'buffer_size': 1000,
    # the top k words to sample from the vocabulary
    'top_k': 5000,
    'learning_rate': 1e-3,
    'n_epochs': 25,
}


def train_captioning_model():
    print('Using hyper-parameters:')
    print(HYPER_PARAMETERS)

    model_save_dir = 'checkpoints/'
    model_name = 'image_captioning_model'

    download_mscoco_data()
    pre_process_images()

    data_generator = load_captioning_training_data(
        batch_size=HYPER_PARAMETERS['batch_size'],
        buffer_size=HYPER_PARAMETERS['buffer_size'],
        top_k=HYPER_PARAMETERS['top_k']
    )

    # Create a distribution strategy to train the model using all the GPUs in
    # one system. If training using multiple systems, use:
    # tf.distribute.experimental.MultiWorkerMirroredStrategy
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = captioning_model.build_captioning_model()

        optimiser = tf.keras.optimizers.Adam(
            lr=HYPER_PARAMETERS['learning_rate']
        )
        # Enable mixed precision using dynamic loss scaling. Combined with XLA,
        # training time should be significantly decreased
        optimiser = tf.train.experimental.enable_mixed_precision_graph_rewrite(
            optimiser)

        model.compile(optimizer=optimiser, loss=captioning_model.loss_function)

        # serialize model to JSON and save
        model_json = model.to_json()
        with open(f'{model_save_dir}{model_name}.json', 'w') as json_file:
            json_file.write(model_json)

    # create callbacks that will be used during model training
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f'tensorboard_logs/{model_name}',
        profile_batch=0,
        write_graph=False
    )

    val_loss_checkpointer = tf.keras.callbacks.ModelCheckpoint(
        f'{model_save_dir}{model_name}_best_val_loss.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=True,
        verbose=True
    )

    loss_checkpointer = tf.keras.callbacks.ModelCheckpoint(
        f'{model_save_dir}{model_name}.h5',
        monitor='loss',
        mode='min',
        save_best_only=False,
        save_weights_only=True,
        verbose=True
    )

    model.fit(
        x=data_generator,
        epochs=HYPER_PARAMETERS['n_epochs'],
        callbacks=[tensorboard, val_loss_checkpointer, loss_checkpointer],
        # The parameters below are auto-tuned by tf.data.Dataset, but can be
        # set manually below.
        # max_queue_size=100,
        # workers=100,
        shuffle=True
    )


if __name__ == '__main__':
    train_captioning_model()
