"""
Ideally, this should be deployed using TensorFlow Serving, where a production
server already has the model loaded into memory and is waiting for API
requests to be made containing images and strings of text. That way, we don't
have to re-load the model every time we want to predict on a new batch of
data, which can be very time consuming.
"""

import click
from difflib import SequenceMatcher
import tensorflow as tf

from utils import load_trained_tf_model, load_captioning_file_paths, \
    compute_features_for_image


@click.command()
@click.argument("image_path", type=str, required=True)
@click.argument("text", type=str, required=True)
@click.argument("captioning_model_save_dir", type=str, required=False,
                default='checkpoints')
@click.argument("captioning_model_name", type=str, required=False,
                default='image_captioning_model')
def predict_on_image_and_text(
        image_path: str,
        text: str,
        captioning_model_save_dir: str,
        captioning_model_name: str,
        top_k: int
) -> float:
    generated_caption = get_caption_from_image(
        image_path, captioning_model_save_dir, captioning_model_name, top_k
    )

    # compute and return a similarity score between 0 and 1 for the texts
    return SequenceMatcher(None, text, generated_caption).ratio()


def get_caption_from_image(
        image_path: str,
        captioning_model_save_dir: str,
        captioning_model_name: str,
        top_k: int,
) -> str:
    try:
        captioning_model = load_trained_tf_model(
            model_save_dir=captioning_model_save_dir,
            model_name=captioning_model_name,
            use_best_val_loss=True
        )
    except FileNotFoundError as e:
        print(e)
        raise ValueError('Could not find a model checkpoint. '
                         'Please train one using train.py first.')

    # Load the tokenizer to decode the model output. We are not interested in
    # the training data returned by the function, so we ignore it.
    _, tokenizer = load_captioning_file_paths(top_k)

    # Pass the image through an InceptionV3 model trained on ImageNet to
    # compute features that will be passed to the captioning model. Please
    # refer to the documentation for utils.pre_process_images to see why this
    # is done.
    image_features, _ = compute_features_for_image(image_path)
    model_output = captioning_model.predict(image_features, batch_size=1)
    result = []

    for char in model_output:
        result.append(tokenizer.index_word[char])

    return ' '.join(result)


if __name__ == '__main__':
    predict_on_image_and_text()
