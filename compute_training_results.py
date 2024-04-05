import math
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tqdm

import torch

from utils import load_embeddings, load_labels, compute_all_labels#, aggregate_embeddings, aggregate_labels

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

def train_model(train_emb:np.ndarray, train_lab:np.ndarray, seed=17) -> torch.nn.Module:
    """
    This function initialize and train a linear model in PyTorch. The model is
    used in a multi-label classification task.
    train_emb is a numpy array (n_frames, emb_dim)
    train_lab is a numpy array (n_frames, n_labels)
    return a torch model that is a simple linear layer trained on train_emb and train_lab
    """
    # shuffle the data
    np.random.seed(seed)
    perm = np.random.permutation(train_emb.shape[0])
    train_emb = train_emb[perm]
    train_lab = train_lab[perm]

    # train
    emb_dim = train_emb.shape[1]
    n_labels = train_lab.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(emb_dim, n_labels),
        torch.nn.Sigmoid()
    )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 100
    for epoch in tqdm.tqdm(range(n_epochs)):
        optimizer.zero_grad()
        outputs = model(torch.tensor(train_emb, dtype=torch.float32))
        loss = criterion(outputs, torch.tensor(train_lab, dtype=torch.float32))
        loss.backward()
        optimizer.step()
    return model


if __name__ == "__main__":
    DATA_FOLDER: str = "data"
    EMB_FOLDER: str = "data/embeddings"
    RESULTS_FOLDER: str = "results"
    IMAGE_FOLDER = "figures"


    embeddings = load_embeddings(EMB_FOLDER)
    labels = load_labels(DATA_FOLDER)
    all_labels = compute_all_labels(labels)

    models = embeddings.keys()
    dataset_names = labels.keys()
    # embeddings is a dict {model: {dataset: list (n_files) of list (n_frames) of numpy arrays (n_features)}}
    # labels is a dict {dataset: list (n_files) of list (n_frames) of list (n_labels)}

    # get indices of data with labels as {dataset: list (n_files) of lists with indices of frames without labels}
    idxs_w_labels = {}
    for dataset in dataset_names:
        idxs_w_labels[dataset] = [
            [i for i, l in enumerate(frame_l) if len(l) != 0]
            for frame_l in labels[dataset]
        ]

    # select embeddings with labels
    embeddings_w_labels = {}
    for model in models:
        emb = embeddings[model]
        new_emb = {}
        for dataset in dataset_names:
            emb_list = emb[dataset] # list (n_files) of numpy arrays (n_frames, n_features)
            idx_list = idxs_w_labels[dataset]
            new_emb[dataset] = [e[i] for e, i in zip(emb_list, idx_list)]
        embeddings_w_labels[model] = new_emb

    # one hot encode labels into {dataset: list (n_files) of np array (n_frames, n_all_labels)}
    # only for the frames that have at least one label
    one_hot_labels = {}
    for dataset in dataset_names:
        lbs = labels[dataset]
        new_lbs = []
        for j, frame_lab_l in enumerate(lbs):
            n_frame_w_lab = len(idxs_w_labels[dataset][j])
            one_hot = np.zeros((n_frame_w_lab, len(all_labels)))
            for i, ind in enumerate(idxs_w_labels[dataset][j]):
                lab_l = frame_lab_l[ind]
                for l in lab_l:
                    one_hot[i, all_labels.index(l)] = 1
            new_lbs.append(one_hot)
        one_hot_labels[dataset] = new_lbs

    # split the data into training and testing
    TRAIN_PERC = 0.75
    splitted_datasets = {m: {d: train_test_split(embeddings_w_labels[m][d], one_hot_labels[d], TRAIN_PERC) for d in dataset_names} for m in models}

    # first, compute the number of elements for each split of each labels
    for model in models:
        for dataset in dataset_names:
            train_emb, train_lab, test_emb, test_lab = splitted_datasets[model][dataset]
            n_train = train_emb.shape[0]
            n_test = test_emb.shape[0]
            print(f"Model {model} on dataset {dataset}")
            print(f"Train: {n_train} frames")
            print(f"Test: {n_test} frames")
            print()
            # 2 columns bar plot of n_frames for each label in train and test
            image_folder = os.path.join(IMAGE_FOLDER, "split_n_frames", model)
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            print("Computing number of frames for each label in train and test")
            print(f"and saving images into {image_folder}")

            train_labels_values = np.sum(train_lab, axis=0)
            test_labels_values = np.sum(test_lab, axis=0)
            n_labels = len(all_labels)
            fig, ax = plt.subplots()
            ind = np.arange(n_labels)
            width = 0.35
            ax.bar(ind, train_labels_values, width, label='Train')
            ax.bar(ind + width, test_labels_values, width, label='Test')
            ax.set_xticks(ind + width / 2)
            ax.set_xticklabels(all_labels, rotation=45)
            ax.legend()
            ax.set_title(f"Model {model} on dataset {dataset}")
            # we will include also the value in the plot
            for i, val in enumerate(train_labels_values):
                ax.text(i, val, str(int(val)), ha='center', va='bottom')
            # save the plot, selecting dpi
            filepath = os.path.join(image_folder, f"{dataset}_split_n_frames.png")
            fig.savefig(filepath, dpi=300)
            plt.close()

    # then, compute for each model for d1 and d2 in dataset, the performance of
    # the model (a simple linear layer) trained on di evaluated on dj
    results = {}
    for model in models:
        results[model] = {}
        for d1 in dataset_names:
            for d2 in dataset_names:
                print(f"Model {model} trained on {d1} and tested on {d2}")
                train_emb, train_lab, _, _ = splitted_datasets[model][d1]
                _, _, test_emb, test_lab = splitted_datasets[model][d2]

                # train a simple linear layer on train_emb and train_lab
                linear_model = train_model(train_emb, train_lab)

                # and evaluate it on test_emb and test_lab
                outputs = linear_model(torch.tensor(test_emb, dtype=torch.float32)).detach().numpy()
                output_classes = np.where(outputs > 0.5, 1, 0)
                results[model][(d1, d2)] = classification_report(output_classes, test_lab, output_dict=True)

    # save the results into a pickle file
    for model in models:
        res_folder = os.path.join(RESULTS_FOLDER, model)
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        with open(os.path.join(res_folder, "training_results.pkl"), "wb") as f:
            pickle.dump(results[model], f)


    # as a benchmark, compute also the performance on the overall dataset