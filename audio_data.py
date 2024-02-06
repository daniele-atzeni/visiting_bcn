import tensorflow as tf
import numpy as np

from utils import load_wav_16k_mono, get_taxonomy_leaves

from yamnet import yamnet_model


class AudioData:
    def __init__(
        self,
        folder: str,
        filename: str,
        year: int,
        labels: dict,
        data: tf.Tensor | None = None,
    ) -> None:
        # init attrs
        self.folder = folder
        self.filename = filename
        self.year = year
        # labels dictionary {n_frame: list of labels}, but it contains empty lists
        # if the labels are all in the background
        self.labels = {k: v for k, v in labels.items() if len(v) > 0}
        self.frames_with_labels = sorted(list(self.labels.keys()))

        self.labels_array = []
        for frame_n in self.frames_with_labels:
            annots = labels[frame_n]
            annot_arr = np.zeros(len(get_taxonomy_leaves()))
            for annot in annots:
                if annot in get_taxonomy_leaves():
                    annot_arr[get_taxonomy_leaves().index(annot)] = 1
            self.labels_array.append(annot_arr)
        self.labels_array = np.array(self.labels_array)

        # save everything only after computing it, unless we are oversampling
        self.data = data
        self.yamnet_output = None
        self.mapped_predictions = None

    def get_data(self) -> tf.Tensor:
        if self.data is None:
            self.data = load_wav_16k_mono(self.folder + self.filename)  # type: ignore problems with the function
        assert self.data is not None, "data is None"
        return self.data

    def get_yamnet_output(self) -> list:
        if self.yamnet_output is None:
            # we don't know for now how to use spectrograms, so we don't save them
            pred, emb, _ = yamnet_model(self.get_data())  # type: ignore problems with the model

            # check if labels and shapes match
            # assert max(self.labels.keys()) < pred.shape[0]
            # I have a case where this fails (index 160 or something like that)
            # I solve it manually
            if max(self.labels.keys()) == pred.shape[0]:
                i = max(self.labels.keys())
                self.labels[i - 1] = self.labels[i]
                self.labels.pop(i)
                self.frames_with_labels.pop(-1)
                self.frames_with_labels.append(i - 1)

            self.yamnet_output = [
                tf.stack([pred[frame] for frame in self.frames_with_labels]),
                tf.stack([emb[frame] for frame in self.frames_with_labels]),
            ]

        return self.yamnet_output

    def get_yamnet_prediction(self) -> tf.Tensor:
        return self.get_yamnet_output()[0]

    def get_yamnet_embeddings(self) -> tf.Tensor:
        return self.get_yamnet_output()[1]
