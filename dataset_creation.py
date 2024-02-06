import os
import numpy as np
import pandas as pd
import librosa
from urllib.request import urlretrieve
from moviepy.editor import VideoFileClip

from tensorflow import Tensor, constant

from utils import create_frame_labels

from audio_data import AudioData


def create_dataset(return_empty_lab_elements=False) -> list:
    labels_folder = "data/labels/"
    labels_file_20 = "SAB-AudioTagging-2020.json"
    labels_frames_20 = create_frame_labels(labels_folder + labels_file_20)

    audios_folder = "data/audios/"
    data_20_folder = audios_folder + "audio_2020/"

    my_dataset = []
    for name in sorted(os.listdir(data_20_folder)):
        if name in labels_frames_20.keys():
            my_dataset.append(
                AudioData(data_20_folder, name, 2020, labels_frames_20[name])
            )

    # 2021
    labels_file_21 = "SAB-AudioTagging-2021.json"
    labels_frames_21 = create_frame_labels(labels_folder + labels_file_21)
    data_21_folder = audios_folder + "audio_2021/"

    for name in sorted(os.listdir(data_21_folder)):
        if name in labels_frames_21.keys():
            my_dataset.append(
                AudioData(data_21_folder, name, 2021, labels_frames_21[name])
            )

    if not return_empty_lab_elements:
        my_dataset = [el for el in my_dataset if len(el.frames_with_labels) > 0]
    return my_dataset


def augment_data(
    audio_data: np.ndarray,
    labels: dict,
    method: str = "noise",
    data_to_add: np.ndarray | None = None,
    label_to_add: dict | None = None,
) -> tuple[Tensor, dict]:
    if method == "noise":
        factor = np.std(audio_data) / np.array(10.0)
        new_audio_data = audio_data + factor * np.random.normal(0, 1, len(audio_data))  # type: ignore this type is wrong
        new_labels = labels
    elif method == "pitch":
        new_audio_data = librosa.effects.pitch_shift(audio_data, sr=16000, n_steps=-5)
        new_labels = labels
    elif method == "time_shift":
        new_audio_data = audio_data
        new_labels = labels
    elif method == "add":
        assert data_to_add is not None and label_to_add is not None
        max_length = max(len(audio_data), len(data_to_add))
        audio_data = np.pad(audio_data, (0, max_length - len(audio_data)))
        data_to_add = np.pad(data_to_add, (0, max_length - len(data_to_add)))
        new_audio_data = audio_data + data_to_add
        new_labels = {}
        for k, v in labels.items():
            if k not in new_labels.keys():
                new_labels[k] = []
            new_labels[k].extend(v)
        for k, v in label_to_add.items():
            if k not in new_labels.keys():
                new_labels[k] = []
            new_labels[k].extend(v)
        new_labels = {k: list(set(v)) for k, v in new_labels.items()}
    else:
        raise ValueError(f"Unknown augmentation method: {method}")
    return constant(new_audio_data, dtype="float32"), new_labels


def create_oversampled_dataset(
    dataset: list[AudioData], min_num_samples=100
) -> list[AudioData]:
    new_dataset = [el for el in dataset]

    labels_count = np.sum(np.concatenate([d.labels_array for d in dataset], 0), 0)
    el_to_add_arr = min_num_samples - labels_count

    for lab_idx, el_to_add in enumerate(el_to_add_arr):
        if el_to_add <= 0:
            continue

        files_with_label = [d for d in dataset if d.labels_array[:, lab_idx].any()]  # type: ignore I know it's a numpy array
        if len(files_with_label) == 0:
            continue
        el_to_add_per_file = int(el_to_add / len(files_with_label))
        for el in files_with_label:
            for i in range(el_to_add_per_file):
                audio_data = el.get_data()
                # create a new audio file from this one
                audio_data = audio_data.numpy()  # type: ignore don't know why
                method = np.random.choice(["noise", "pitch", "time_shift", "add"])
                if method == "add":
                    data_to_add_idx = np.random.choice(
                        [idx for idx in range(len(dataset)) if idx != el]
                    )
                    data_to_add = dataset[data_to_add_idx].get_data().numpy()
                    label_to_add = dataset[data_to_add_idx].labels
                else:
                    data_to_add, label_to_add = None, None

                new_audio, new_lab = augment_data(
                    audio_data,
                    el.labels,
                    method=method,
                    data_to_add=data_to_add,
                    label_to_add=label_to_add,
                )
                # init new AudioData
                new_el = AudioData(
                    el.folder,
                    f"overs_{i}_" + el.filename,
                    el.year,
                    new_lab,
                    new_audio,
                )
                new_dataset.append(new_el)

    return new_dataset


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
