import os
import json
import pickle

import numpy as np
from tensorflow.data import TFRecordDataset
from perch.chirp.inference import tf_examples
from etils import epath

from utils_new import create_frame_labels


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
            )  # type:ignore   not my code
            embeddings.append(embedding)
        return embeddings
    if model == "yamnet":
        # in this case the embedding is a list of numpy arrays
        # (n_audios=64, max_n_frames=148, embedding_dim=1024)
        filename = os.listdir(folder)[0]
        with open(os.path.join(folder, filename), "rb") as f:
            embeddings = pickle.load(f)
        return [e.numpy() for e in embeddings]
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


main_folder = "data/embeddings"
RESULTS_FOLDER = "data/processed_embeddings"
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

labels_2020_filename = "data/sons_al_balco_labels/SAB-AudioTagging-2020.json"
labels_2021_filename = "data/sons_al_balco_labels/SAB-AudioTagging-2021.json"

for model in os.listdir(main_folder):
    orig_model_folder = os.path.join(main_folder, model)
    res_model_folder = os.path.join(RESULTS_FOLDER, model)
    if not os.path.exists(res_model_folder):
        os.makedirs(res_model_folder)

    # split between configurations and datasets
    with open(os.path.join(orig_model_folder, "config.json"), "r") as f:
        config = json.load(f)  # this contains window_size and windows_shift
    datasets = os.listdir(orig_model_folder)
    assert "config.json" in datasets, f"config.json not found for {model}"
    datasets.pop(datasets.index("config.json"))

    for dataset in datasets:
        orig_dataset_folder = os.path.join(orig_model_folder, dataset)
        res_dataset_folder = os.path.join(res_model_folder, dataset)
        if not os.path.exists(res_dataset_folder):
            os.makedirs(res_dataset_folder)

        # let's load the embeddings as a list of np.arrays
        print(f"Modifying {model} {dataset} embeddings...")
        embeddings = load_embeddings(orig_dataset_folder, model)
        if model == "perch":
            # embeddings are 3D, we must remove the middle one
            embeddings = [emb.squeeze(axis=1) for emb in embeddings]
        # save the results with pickle
        print("Saving embeddings...")
        with open(os.path.join(res_dataset_folder, "embeddings.pkl"), "wb") as f:
            pickle.dump(embeddings, f)
        print("Done")

        # modify labels if dataset is sons al balco
        if "sons_al_balco" not in dataset:
            pass
        else:
            if dataset == "sons_al_balco_2020":
                labels = create_frame_labels(
                    labels_2020_filename, config["window_size"], config["window_shift"]
                )
            if dataset == "sons_al_balco_2021":
                labels = create_frame_labels(
                    labels_2021_filename, config["window_size"], config["window_shift"]
                )

            # check if labels length is the same of the embeddings, if not append
            # empty lists
            assert len(labels) == len(
                embeddings
            ), "Embeddings and labels have different lengths"
            new_labels = []
            for i, (emb, lab) in enumerate(zip(embeddings, labels)):
                if len(lab) > emb.shape[0]:
                    lab = lab[: emb.shape[0]]
                else:
                    el_to_add = emb.shape[0] - len(lab)
                    for _ in range(el_to_add):
                        lab.append([])
                new_labels.append(lab)

            print("Saving labels...")
            with open(os.path.join(res_dataset_folder, "labels.pkl"), "wb") as f:
                pickle.dump(new_labels, f)
            print("Done")
