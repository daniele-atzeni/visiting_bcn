import os
import json
import math

import numpy as np
import pandas as pd
import librosa
from urllib.request import urlretrieve
from moviepy.editor import VideoFileClip

from tensorflow.data import TFRecordDataset

from perch.chirp.inference import tf_examples
from etils import epath


def create_audio_data_from_videos(filename: str, pre_url: str) -> list:
    """
    This function will create a list of audios from the csv containing the links
    of the videos, as well as other information that we still don't know how to
    use.
    """
    df = pd.read_excel(filename)
    # filenames are in column "videoURL"
    urls = df["videoURL"].values
    audios = []
    for i, url in enumerate(urls):
        try:
            # download the video
            # url = "https://enquestes.salle.url.edu/videos_sons/granollers/video_626e7fa5028975.32983894.mp4"
            tmp_video_name = "tmp_video.mp4"
            urlretrieve(pre_url + url, tmp_video_name)
            # load the file with moviepy
            clip = VideoFileClip(tmp_video_name)
            assert clip.audio is not None
            # save the audio to a temp file
            tmp_audio_name = "tmp_audio.mp3"
            clip.audio.write_audiofile(tmp_audio_name)
            # reload it with librosa
            audio_data = librosa.load(tmp_audio_name)
            # remove the files
            os.remove(tmp_video_name)
            os.remove(tmp_audio_name)
            # append to the list
            audios.append(audio_data)
        except OSError:
            print(f"Error with video {i}")
    return audios


def create_frame_labels(
    labels_filename: str, frame_length:float = 0.96, frame_shift: float = 0.48
) -> list[list[list[str]]]:
    """
    This function create a dictionary {filename: {n_frame: list of labels}}
    n_frame refers to the frame number in the original file, considering the given
    shift. Then we will modify it to make it a list[list[list]] of labels, for
    each label of each frame of each file.

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

    res = []
    for _, labels in sorted(foreground_lab.items()):
        res.append([labels.get(i, []) for i in range(max(labels.keys()) + 1)])

    return res


def load_embeddings(folder, model) -> list[np.ndarray]:
    """
    This function returns a list of embeddings, one for each audio in the folder.
    The embeddings are numpy arrays of shape (n_frames, embedding_dim). The number
    of frames depends on the pre-trained model used and its configuration.
    """

    if model == "birdnet":
        # in this case the embedding is a text file for each audio
        # the file is start_time \t end_time \t embedding \n
        # embedding is comma separated float values
        # frames are 3s long (possibly configurable)
        embeddings = []
        for filename in sorted(os.listdir(folder)):
            with open(os.path.join(folder, filename), "r") as f:
                lines = f.readlines()
            # first two els are start and end time
            embedding = np.array(
                [list(map(float, line.split("\t")[2].split(","))) for line in lines]
            )
            embeddings.append(embedding)
        return embeddings
    if model == "yamnet":
        # in this case the embedding is a numpy array
        # (n_audios=64, max_n_frames=148, embedding_dim=1024)
        # the padding value is 0, the frames are every 0.48 (probably)
        filename = os.listdir(folder)[0]
        embedding = np.load(os.path.join(folder, filename))
        return [emb for emb in embedding]
    if model == "perch":
        # in this case the results are written with tensorflow TFRecordWriter
        # The way to load them is in perch/embed_audio.ipynb
        output_dir = epath.Path(folder)
        fns = [fn for fn in output_dir.glob("embeddings-*")]
        ds = TFRecordDataset(fns)
        parser = tf_examples.get_example_parser()
        ds = ds.map(parser)
        res_dict = {}
        for ex in ds.as_numpy_iterator():
            res_dict[ex["filename"]] = ex["embedding"]
        res = [v for _, v in sorted(res_dict.items())]
        return res

    else:
        raise ValueError(f"Unknown pretrained model {model}")


def modify_embedding_windows(
    original_embeddings: list[np.ndarray],
    original_w_size: float,
    original_w_shift: float,
    new_w_size: float,
    aggregation: str = "mean",
) -> list[np.ndarray]:
    """
    This function modify a dataset of embeddings and labels (i.e. a list of
    embeddings of shape (n_frames, embedding_dim) and a list of list of str) in
    order to have embeddings corresponding to other window sizes. It needs:
    - the original embeddings
    - the original labels
    - the original window size
    - the original window shift (the step between two consecutive windows)
    - the new window size (not all values are ok)
    - an aggregation function (e.g. mean, max, min)
    """

    assert (
        new_w_size >= original_w_size + original_w_shift
    ), "The new window size must be greater than the original window size"
    assert_cond = (
        new_w_size % original_w_size == 0
        if original_w_shift == 0
        else (new_w_size - original_w_size) % original_w_shift == 0
    )
    assert (
        assert_cond
    ), "The new window size must be a multiple of the original window shift (or window size if shift is 0)"

    assert aggregation in [
        "mean",
        "sum",
        "max",
        "min",
    ], "Aggregation function not recognized"

    def aggr_emb_fn(l):
        if aggregation == "sum":
            return np.sum(l, axis=0)
        elif aggregation == "mean":
            return np.mean(l, axis=0)
        elif aggregation == "max":
            return np.max(l, axis=0)
        # elif aggregation == "min":
        return np.min(l, axis=0)

    new_embeddings = []
    for emb in original_embeddings:
        if original_w_shift == 0:
            n_emb = int(new_w_size / original_w_size)
        else:
            n_emb = int((new_w_size - original_w_size) / original_w_shift)
        if len(emb.shape) > 2:
            emb = emb.reshape(-1, emb.shape[-1])
        start = 0
        new_emb = []
        while start < emb.shape[0]:
            end = start + n_emb
            new_emb.append(aggr_emb_fn(emb[start:end]))
            start = end
        new_embeddings.append(np.stack(new_emb, axis=0))

    return new_embeddings


def modify_yamnet_labels_windows(
    orig_labels: dict,
    new_w_size: float,
    original_w_size: float = 1,
    original_w_shift: float = 0.5,
) -> list[list[str]]:
    """
    Modify the labels created by create_yamnet_frame_labels in order to return a
    list of lists of labels, one for each frame of the model.
    The frame window is specified by the new_window_size parameter, while the original
    values are the ones from Yamnet (window_size=0.96 and window_shift=0.48). We
    will use 0.5 and 1 to ease the computation with the other embeddings.
    The files are sorted in ascending order.
    """

    assert (
        new_w_size >= original_w_size + original_w_shift
    ), "The new window size must be greater than the original window size"
    assert_cond = (
        new_w_size % original_w_size == 0
        if original_w_shift == 0
        else (new_w_size - original_w_size) % original_w_shift == 0
    )
    assert (
        assert_cond
    ), "The new window size must be a multiple of the original window shift (or window size if shift is 0)"

    def aggr_lab_fn(l):
        res = []
        for el in l:
            res.extend(el)
        return list(set(res))

    new_labels = []
    if original_w_shift == 0:
        n_emb = int(new_w_size / original_w_size)
    else:
        n_emb = int((new_w_size - original_w_size) / original_w_shift)
    start = 0
    while start < max(orig_labels.keys()):
        end = start + n_emb
        lab_list = [orig_labels.get(i, []) for i in range(start, end)]
        new_labels.append(aggr_lab_fn(lab_list[start:end]))
        start = end

    return new_labels
