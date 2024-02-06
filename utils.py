import json
import math

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_io as tfio

from taxonomy_leaves import taxonomy_leaves

from yamnet import *


@tf.function
def load_wav_16k_mono(filename: str) -> tf.Tensor:
    """Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio."""
    tf.compat.v1.logging.set_verbosity(
        tf.compat.v1.logging.ERROR
    )  # this for not uotputting warnings

    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)  # type: ignore not my code
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav


def get_yamnet_classes() -> list:
    class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")  # type: ignore not my code
    return list(pd.read_csv(class_map_path)["display_name"])


def get_taxonomy_leaves() -> list:
    return sorted(taxonomy_leaves)


def create_frame_labels(labels_filename: str, frame_shift: float = 0.48) -> dict:
    """
    This function create a dictionary {filename: {n_frame: list of labels}}
    n_frame refers to the frame number in the original file, considering the given
    shift.

    Original labels contains a list of annotations and other important fields.
    We will use only foreground labels.

    Regarding Yamnet, it automatically splits the audio file in frames of 0.96s
    with a shift of 0.48s
    """
    with open(labels_filename) as f:
        original_labels = json.load(f)

    frame_lab = {}
    for lab in original_labels:
        name = lab["original_filename"]
        # result will be a dictionary {n_frame: list of labels} because idk the length of the audio file
        frame_lab[name] = {}
        for annot in lab["segmentations"]:
            start = annot["start_time"]
            end = annot["end_time"]
            # the real annotation is in annot['annotations']
            annot = annot["annotations"]
            act_lab = annot["Element"]["values"][0]["value"]
            try:
                dist = annot["Distance"]["values"][0]["value"]
            except KeyError:
                dist = "Unknown"
            # compute first and last frames that contain this annotation
            first_frame = math.floor(start / frame_shift)
            last_frame = max(first_frame, math.floor(end / frame_shift) - 1)
            ### there is a problem if the annotation end when the video ends
            for i in range(last_frame - first_frame + 1):
                if first_frame + i not in frame_lab[name].keys():
                    frame_lab[name][first_frame + i] = []
                frame_lab[name][first_frame + i].append((act_lab, dist))

    # let's take only foreground labels
    foreground_lab = {}
    for name, lab in frame_lab.items():
        foreground_lab[name] = {}
        for n_frame, annot in lab.items():
            foreground_lab[name][n_frame] = [
                lab for lab, dist in annot if dist == "Foreground"
            ]

    return foreground_lab


def split_train_test(
    dataset_x: list, dataset_y: list, test_perc: float = 0.2
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    # dataset_x is a list of tensors (embeddings)
    # dataset_y is a list of numpy arrays (one-hot encoded labels)
    label_names = get_taxonomy_leaves()

    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for emb, lab in zip(dataset_x, dataset_y):
        final_tr_ind = math.floor(emb.shape[0] * (1 - test_perc))
        train_x.append(emb[:final_tr_ind])
        train_y.append(tf.constant(lab[:final_tr_ind]))

        # +1 because we want to avoid frames intersections
        test_x.append(emb[final_tr_ind + 1 :])
        test_y.append(tf.constant(lab[final_tr_ind + 1 :]))

    train_x = tf.concat(train_x, 0)
    train_y = tf.concat(train_y, 0)
    test_x = tf.concat(test_x, 0)
    test_y = tf.concat(test_y, 0)

    return train_x, train_y, test_x, test_y  # type: ignore tf.concat returns a tensor


def map_predictions(
    pred_list: list[np.ndarray], taxonomy_map: dict | list
) -> list[np.ndarray]:
    """
    This function maps prediction probabilities with certain labels into predictions
    with other labels, according to the taxonomy map.
    pred_list is a list of numpy arrays (n_frames_with_labels, n_classes).
    We will use max to map predictions.
    """
    n_mapped_labels = (
        max(taxonomy_map.values())
        if isinstance(taxonomy_map, dict)
        else max(taxonomy_map)
    ) + 1

    mapped_pred_list = []
    for pred in pred_list:
        n_el, _ = pred.shape

        mapped_pred = np.zeros((n_el, n_mapped_labels))
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                mapped_pred[i, taxonomy_map[j]] = max(
                    mapped_pred[i, taxonomy_map[j]], pred[i, j]
                )

        mapped_pred_list.append(mapped_pred)
    return mapped_pred_list
