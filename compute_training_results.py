import math

import numpy as np

from utils import load_per_file_processed_data

def train_test_split(embeddings:list[np.ndarray], labels:list[np.ndarray], train_perc:float=0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    embeddings is a list (n_files) of numpy arrays (n_frames, n_features)
    labels is a list (n_files) of numpy arrays (n_frames, n_labels)
    train_perc is the percentage of data to use for training.
    return a tuple of 4 numpy arrays: train_emb, train_lab, test_emb, test_lab of shape
    ( n_(train/test)_frames, emb_dim/n_lab )
    """
    num_train = [math.floor(emb.shape[0] * train_perc) for emb in embeddings]
    train_emb = [emb[:n] for emb, n in zip(embeddings, num_train)]
    train_lab = [lab[:n] for lab, n in zip(labels, num_train)]
    test_emb = [emb[n:] for emb, n in zip(embeddings, num_train)]
    test_lab = [lab[n:] for lab, n in zip(labels, num_train)]

    train_emb = np.concatenate(train_emb, axis=0)
    train_lab = np.concatenate(train_lab, axis=0)
    test_emb = np.concatenate(test_emb, axis=0)
    test_lab = np.concatenate(test_lab, axis=0)

    return train_emb, train_lab, test_emb, test_lab


DATA_FOLDER = "data/processed_embeddings"
embeddings, labels, all_labels = load_per_file_processed_data(DATA_FOLDER)

# remove embeddings that does not contain any label
for k in embeddings.keys():
    # embeddings and labels share the same keys
    embs = embeddings[k]
    lbs = labels[k]
    # they are lists of (n_files) elements
    idxs = [[i for i, l in enumerate(frame_l) if len(l) > 0] for frame_l in lbs]

    embeddings[k] = [[emb_l[i] for i in idx_l] for emb_l, idx_l in zip(embs, idxs) if len(idx_l) > 0]
    labels[k] = [[lab_l[i] for i in idx_l] for lab_l, idx_l in zip(lbs, idxs) if len(idx_l) > 0]

# convert labels into one hot encoding
for k in labels.keys():
    lbs = labels[k] # n_file * n_frame * n_labels
    new_lbs = []
    for frame_lab_l in lbs:
        one_hot = np.zeros((len(frame_lab_l), len(all_labels)))
        for i, lab_l in enumerate(frame_lab_l):
            for l in lab_l:
                one_hot[i, all_labels.index(l)] = 1
        new_lbs.append(one_hot)
    labels[k] = new_lbs

# convert list of embeddings into numpy arrays (now they are list of 1D arrays)
for k in embeddings.keys():
    all_emb = []
    for frame_l in embeddings[k]:
        all_emb.append(np.stack(frame_l))
    embeddings[k] = all_emb

# split the data into training and testing
TRAIN_PERC = 0.8
splitted_datasets = {k: train_test_split(embeddings[k], labels[k], TRAIN_PERC) for k in embeddings.keys()}

models = sorted(set([e for e, _ in splitted_datasets.keys()]))
dataset_names = sorted(set([e for _, e in splitted_datasets.keys()]))

# first, compute the number of elements for each split of each labels

# then, compute for each model for d1 and d2 in dataset, the performance of
# the model (a simple linear layer) trained on d1 evaluated on d2

# as a benchmark, compute also the performance on the overall dataset