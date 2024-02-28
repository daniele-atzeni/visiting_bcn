import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Number of colors
num_colors = 40

# List of qualitative colormaps
cmaps = [
    "tab10",
    "tab20",
    "Set3",
    "Set2",
    "Set1",
    "Pastel1",
    "Pastel2",
    "Dark2",
    "Accent",
]

# Generate colors
colors = []
for cmap_name in cmaps:
    cmap = mpl.colormaps.get_cmap(cmap_name)
    colors.extend(cmap(np.arange(cmap.N)))

# If we have more colors than needed, truncate the list
if len(colors) > num_colors:
    colors = colors[:num_colors]


def plot_overall_embeddings(
    data,
    labels,
    idxs,
    datasets,
    all_labels,
    model,
    image_folder,
    size: int = 3,
    alpha: float = 0.6,
):
    n_col = len(datasets) // 2 + len(datasets) % 2
    fig, axs = plt.subplots(n_col, 2, figsize=(20, 10))

    for i, dataset in enumerate(datasets):
        start, end = idxs[dataset]
        data_subset = data[start:end]
        lab_list_subset = labels[start:end]

        # select points with at least one label
        idx = [i for i, l in enumerate(lab_list_subset) if len(l) > 0]
        data_subset = data_subset[idx]
        lab_list_subset = [lab_list_subset[i] for i in idx]

        for j, lab in enumerate(all_labels):
            idx = [k for k, l in enumerate(lab_list_subset) if lab in l]
            axs[i].scatter(
                x=data_subset[idx, 0],
                y=data_subset[idx, 1],
                label=lab,
                color=colors[j],
                s=size,
                alpha=alpha,
            )
        axs[i].set_title(dataset.replace("_", " ").title())

    plt.suptitle(f"{model} embeddings")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(image_folder, f"{model}_all_datasets.png"), dpi=300)
    plt.close()


def plot_labels_embeddings(
    data,
    labels,
    idxs,
    datasets,
    all_labels,
    model,
    image_folder,
    size: int = 5,
    alpha: float = 0.6,
):

    for lab in all_labels:
        plt.figure(figsize=(15, 15))

        for i, dataset in enumerate(datasets):
            start, end = idxs[dataset]
            data_subset = data[start:end]
            lab_list_subset = labels[start:end]

            # select points with at least one label
            idx = [j for j, l in enumerate(lab_list_subset) if lab in l]
            data_subset = data_subset[idx]
            plt.scatter(
                x=data_subset[:, 0],
                y=data_subset[:, 1],
                label=dataset.replace("_", " ").title(),
                color=colors[i],
                s=size,
                alpha=alpha,
            )

        plt.title(f"{model} {lab} embeddings")
        plt.legend()
        filename = f"{model}_{lab}_all_datasets.png".replace("/", "_")
        plt.savefig(os.path.join(image_folder, filename), dpi=300)
        plt.close()


folder = "data/processed_embeddings"
IMAGE_FOLDER = "figures/2d_plots"
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# datasets = ["sons_al_balco_2020", "sons_al_balco_2021", "sabadell", "granollers"]
datasets = ["sons_al_balco_2020", "sons_al_balco_2021"]
models = ["yamnet"]  # ["perch", "birdnet", "yamnet"]
for model in models:
    print(f"Plotting for model {model}")
    image_folder = os.path.join(IMAGE_FOLDER, model)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    embeddings = {}
    for dataset in datasets:
        # load embeddings and labels with pickle
        with open(os.path.join(folder, model, dataset, "embeddings.pkl"), "rb") as f:
            emb = pickle.load(f)
        with open(os.path.join(folder, model, dataset, "labels.pkl"), "rb") as f:
            lab = pickle.load(f)
        embeddings[dataset] = (emb, lab)

    # we will first reduce dimensionality with pca to speed up computations
    data = np.array([])
    for _, (emb_list, _) in sorted(embeddings.items()):
        if len(data) == 0:
            data = np.concatenate(emb_list)
        else:
            data = np.concatenate([data] + emb_list)

    pca = PCA(n_components=50)
    pca_data = pca.fit_transform(data)
    # compute tsne
    BEST_PERPL = 30
    BEST_EARLY_EX = 12
    tsne = TSNE(n_components=2, perplexity=BEST_PERPL, early_exaggeration=BEST_EARLY_EX)
    tsne_data = tsne.fit_transform(pca_data)
    # compute umap
    BEST_N_NEIGH = 15
    BEST_MIN_DIST = 0.1
    # umap = umap.UMAP(n_neighbors=BEST_N_NEIGH, min_dist=BEST_MIN_DIST)
    # umap_data = umap.fit_transform(pca_data)

    # convert labels into list of lists, as data
    labels = []
    for _, (_, file_lab_list) in sorted(embeddings.items()):
        # list (n_files) of lists (n_frame) of labels
        for frame_lab_list in file_lab_list:
            # list of lists of labels (len = n_frames_per_file)
            labels.extend(frame_lab_list)

    # retrieve the start and end index for each dataset
    idxs = {}
    start_idx = 0
    end_idx = 0
    for dataset, (emb_list, _) in sorted(embeddings.items()):
        end_idx += sum([e.shape[0] for e in emb_list])
        idxs[dataset] = (start_idx, end_idx)
        start_idx = end_idx
    # check correctness
    assert end_idx == data.shape[0]

    # plot tsne and umap overall with colors == labels and for each label
    all_labels = tuple(set([l for lab in labels for l in lab]))

    plot_overall_embeddings(
        tsne_data, labels, idxs, datasets, all_labels, "tsne", image_folder
    )
    # plot_overall_embeddings(umap_data, labels, idxs, datasets, all_labels, "umap")

    plot_labels_embeddings(
        tsne_data, labels, idxs, datasets, all_labels, "tsne", image_folder
    )
    # plot_labels_embeddings(umap_data, labels, idxs, datasets, all_labels, "umap")
