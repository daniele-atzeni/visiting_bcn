import os
import json
import math
import pickle

import numpy as np
import pandas as pd
import librosa
from urllib.request import urlretrieve
from moviepy.editor import VideoFileClip

from tensorflow.data import TFRecordDataset

#from perch.chirp.inference import tf_examples
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
    labels_filename: str, frame_length: float = 0.96, frame_shift: float = 0.48
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


def load_joint_processed_data(data_folder: str) -> tuple[dict, dict, list]:
    """
    This function load embeddings as numpy arrays insted of a list of numpy arrays
    for each file. It does the same for the labels, obtaining a list (for each frame)
    of lists of labels.
    The computation is done for all the models and datasets and returned as a dict
    (model, dataset_name) : embeddings/labels. It also returns the ordered tuple
    of unique labels.
    """

    models = os.listdir(data_folder)    # type:ignore it works with string

    dataset_names = [
        "sons_al_balco_2020",
        "sons_al_balco_2021",
    ]  # os.listdir(os.path.join(data_folder, models[0]))

    embeddings_filename = "embeddings.pkl"
    labels_filename = "labels.pkl"

    embeddings = {}
    for model in models:
        for dataset in dataset_names:
            with open(
                os.path.join(data_folder, model, dataset, embeddings_filename), "rb"
            ) as f:
                emb_list = pickle.load(f)

            embeddings[(model, dataset)] = np.concatenate(emb_list)

    labels = {}
    for model in models:
        for dataset in dataset_names:
            with open(
                os.path.join(data_folder, model, dataset, labels_filename), "rb"
            ) as f:
                lab_list = pickle.load(f)
            labels[(model, dataset)] = []
            for file_list in lab_list:
                labels[(model, dataset)].extend(file_list)

    all_labels = []
    # labels values are lists (n_file) of lists (n_frame) of lists (n_labels)
    for frame_l in labels.values():
        for lab_l in frame_l:
            all_labels.extend(lab_l)
    all_labels = sorted(set(all_labels))

    return embeddings, labels, all_labels


def load_per_file_processed_data(data_folder: str) -> tuple[dict, dict, list]:
    """
    This function load embeddings as lists of numpy arrays for each file. 
    It does the same for the labels, obtaining a list (for each frame)
    of lists of labels.
    The computation is done for all the models and datasets and returned as a dict
    (model, dataset_name) : embeddings/labels. It also returns the ordered tuple
    of unique labels.
    """

    models = os.listdir(data_folder)    # type:ignore it works with string

    dataset_names = [
        "sons_al_balco_2020",
        "sons_al_balco_2021",
    ]  # os.listdir(os.path.join(data_folder, models[0]))

    embeddings_filename = "embeddings.pkl"
    labels_filename = "labels.pkl"

    embeddings = {}
    for model in models:
        for dataset in dataset_names:
            with open(
                os.path.join(data_folder, model, dataset, embeddings_filename), "rb"
            ) as f:
                emb = pickle.load(f)  # it's a list of list of arrays
                embeddings[(model, dataset)] = [np.stack(e) for e in emb]

    labels = {}
    for model in models:
        for dataset in dataset_names:
            with open(
                os.path.join(data_folder, model, dataset, labels_filename), "rb"
            ) as f:
                labels[(model, dataset)] = pickle.load(f)

    all_labels = []
    # labels values are lists (n_file) of lists (n_frame) of lists (n_labels)
    for file_l in labels.values():
        for frame_l in file_l:
            for lab_l in frame_l:
                all_labels.extend(lab_l)
    all_labels = sorted(set(all_labels))

    return embeddings, labels, all_labels


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





def load_embeddings(folder:str) -> dict[str, dict[str, list[np.ndarray]]]:
    """
    Function to load embeddings from YamNet, OpenL3, and wav2vec models.
    In this case, folders are organized as 
    folder
        model1
            dataset1
                file1
                file2
                ...
            dataset2
                file1
                file2
                ...
            ...
        model2
            ...
        ...
    
    File in this case are pickle that contains numpy arrays (n_frames, emb_dim).
    Each frame correspond to more or less a second of audio.
    Results are {model : {dataset : list of numpy arrays}}.
    Each element of the list correspond to a file (sorted in alphabetical order).
    """
    models = os.listdir(folder)
    datasets = os.listdir(os.path.join(folder, models[0]))

    results = {}
    for model in models:
        results[model] = {}
        for dataset in datasets:
            path = os.path.join(folder, model, dataset)
            arr_list = []
            for filename in sorted(os.listdir(path)):
                with open(os.path.join(path, filename), "rb") as f:
                    arr_list.append(pickle.load(f))
            results[model][dataset] = arr_list.copy()
    
    return results


def load_sons_al_balco_labels(root_folder:str) -> dict[str, dict[str, list[list[str]]]]:
    """
    This function loads the labels of the Sons al Balco dataset for Yamnet, OpenL3 and 
    wav2vec models. We assume that the embeddings are already computed, so we use them
    to understand how many frames are in each file, hence we can decide in which frame
    we must include each label.
    The returned structure is the same as the one of load_embeddings without models hierarchy, 
    except that it returns list of labels instead of numpy arrays, then:
    {dataset : list (n_files) of lists (n_frames in file) of lists (n_labels in frame)}
    """
    embeddings = load_embeddings(os.path.join(root_folder, "embeddings"))
    n_frames_dict = {}
    emb_dict = list(embeddings.values())[0] # n_frames is independent from the model
    for dataset in emb_dict.keys():
        n_frames_dict[dataset] = [e.shape[0] for e in emb_dict[dataset]]
        # embeddings are ordered by filename

    datasets = ["sons_al_balco_2020", "sons_al_balco_2021"]
    
    # get the length of the files
    files_length = {}
    for dataset in datasets:
        files_length[dataset] = {}
        filenames = sorted(os.listdir(os.path.join(root_folder, f"{dataset}_audios")))
        filenames = [f for f in filenames if not f.endswith(".DS_Store")]
        for i, el in enumerate(filenames):
            audio_data, sr = librosa.load(os.path.join(root_folder, f"{dataset}_audios", el))
            files_length[dataset][i] = len(audio_data) / sr

    labels = {}
    for dataset in datasets:
        labels[dataset] = []    # list of files
        filename = os.path.join(root_folder, f"{dataset}_labels", "labels.json")
        with open(filename, "r") as f:
            labels_list = json.load(f)
        sorted_labels = sorted(labels_list, key=lambda x: x["original_filename"])
    
        for i, el in enumerate(sorted_labels):
            file_length = files_length[dataset][i]
            n_frames = n_frames_dict[dataset][i]
            frames_list = [[] for _ in range(n_frames)]
            # el["segmentations"] is a list of dictionaries
            # in these dictionaries we care about "start_time", "end_time" and "annotations"
            # annotations is a dictionary with "Distance", that contains foreground/background
            # in ["Distance"]["values"][0]["value"], and "Element" that contains the label
            # in ["Element"]["values"][0]["value"]
            for annot in el["segmentations"]:
                start = annot["start_time"]
                end = annot["end_time"]
                act_lab = annot["annotations"]["Element"]["values"][0]["value"]
                if "Distance" not in annot["annotations"]:
                    dist = "Foreground"
                else:
                    dist = annot["annotations"]["Distance"]["values"][0]["value"]
                if dist != "Foreground":
                    continue
                # compute first and last frames that contain this annotation
                frame_length = file_length / n_frames
                first_frame = math.floor(start / frame_length)
                last_frame = max(first_frame, math.floor(end / frame_length) - 1)
                for fr in range(first_frame, last_frame + 1):
                    frames_list[fr].append(act_lab)
            labels[dataset].append(frames_list)            

    return labels


def load_granollers_labels(root_folder:str) -> dict[str, dict[str, list[list[str]]]]:
    """
    This function loads the labels of the Granollers dataset for Yamnet, OpenL3 and 
    wav2vec models. We assume that the embeddings are already computed, so we use them
    to understand how many frames are in each file, hence we can decide in which frame
    we must include each label.
    The returned structure is the same as the one of load_embeddings without models hierarchy, 
    except that it returns list of labels instead of numpy arrays, then:
    {dataset : list (n_files) of lists (n_frames in file) of lists (n_labels in frame)}
    """
    embeddings = load_embeddings(os.path.join(root_folder, "embeddings"))
    n_frames_dict = {}
    emb_dict = list(embeddings.values())[0] # n_frames is independent from the model
    for dataset in emb_dict.keys():
        n_frames_dict[dataset] = [e.shape[0] for e in emb_dict[dataset]]
        # embeddings are ordered by filename

    datasets = ["granollers"]
    
    # get the length of the files
    files_length = {}
    for dataset in datasets:
        files_length[dataset] = {}
        filenames = sorted(os.listdir(os.path.join(root_folder, f"{dataset}_audios")))
        filenames = [f for f in filenames if not f.endswith(".DS_Store")]
        for i, el in enumerate(filenames):
            audio_data, sr = librosa.load(os.path.join(root_folder, f"{dataset}_audios", el))
            files_length[dataset][i] = len(audio_data) / sr

    labels = {}
    for dataset in datasets:
        labels[dataset] = []    # list of files
        label_path = os.path.join(root_folder, f"{dataset}_labels")
        # in this case the files are text files with one row for each label
        # each row is start_time \t end_time \t label
        for i, filename in enumerate(sorted(os.listdir(label_path))):
            filepath = os.path.join(label_path, filename)
            with open(filepath, "r") as f:
                lines = f.readlines()
            frames_list = [[] for _ in range(n_frames_dict[dataset][i])]
            for line in lines:
                try:
                    start, end, label = line.strip().split("\t")
                except ValueError:
                    print(f"Error in labeling file {filename} with line {line}")
                    continue
                start = float(start)
                end = float(end)
                # check for start and end time
                start = max(start, 0)
                end = min(end, files_length[dataset][i])

                frame_length = files_length[dataset][i] / n_frames_dict[dataset][i]
                first_frame = math.floor(start / frame_length)
                last_frame = max(first_frame, math.floor(end / frame_length) - 1)
                for fr in range(first_frame, last_frame + 1):
                    frames_list[fr].append(label)
            labels[dataset].append(frames_list)
    
    return labels


def load_labels(root_folder:str) -> dict[str, dict[str, list[list[str]]]]:
    sons_al_balco_labels = load_sons_al_balco_labels(root_folder)
    granollers_labels = load_granollers_labels(root_folder)
    res = {**sons_al_balco_labels, **granollers_labels}
    return {**sons_al_balco_labels, **granollers_labels}


def compute_all_labels(labels) -> list[str]:
    """
    This function returns the list of all the labels present in the dataset.
    """
    all_labels = []
    for dataset in labels.values():
        for file in dataset:
            for frame in file:
                all_labels.extend(frame)
    return sorted(set(all_labels))

# let's modify embeddings to be {model: {dataset: np.array (total_frames, features)}}
# and labels to be {dataset: [list (tot n frames) of lists (labels in frame)]}

def aggregate_embeddings(embeddings) -> dict[str, dict[str, np.ndarray]]:
    """
    This function aggregates embeddings in order to have a dictionary with the
    structure {model: {dataset: np.array (total_frames, features)}}.
    """
    res = {}
    for model, dataset_dict in embeddings.items():
        res[model] = {}
        for dataset, emb_list in dataset_dict.items():
            res[model][dataset] = np.concatenate(emb_list)
    return res

def aggregate_labels(labels) -> dict[str, list[list[str]]]:
    """
    This function aggregates labels in order to have a dictionary with the
    structure {dataset: [list (tot n frames) of lists (labels in frame)]}.
    """
    res = {}
    for dataset, file_list in labels.items():
        res[dataset] = []
        for file in file_list:
            res[dataset].extend(file)
    return res