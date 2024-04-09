import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from utils import load_joint_processed_data

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
    filename: str = "all_datasets",
):
    n_col = len(datasets) // 2 + len(datasets) % 2
    _, axs = plt.subplots(n_col, 2, figsize=(20, 10))

    for i, dataset in enumerate(datasets):
        start, end = idxs[dataset]
        data_subset = data[start:end]
        lab_list_subset = labels[start:end]

        # select points with at least one label
        # idx = [l_i for l_i, l in enumerate(lab_list_subset) if len(l) > 0]
        # data_subset = data_subset[idx]
        # lab_list_subset = [lab_list_subset[k] for k in idx]

        for j, lab in enumerate(all_labels):
            lab_idx = [l_i for l_i, l in enumerate(lab_list_subset) if lab in l]
            if len(lab_idx) == 0:
                continue
            axs[i].scatter(
                x=data_subset[lab_idx, 0],
                y=data_subset[lab_idx, 1],
                label=lab,
                color=colors[j],
                s=size,
                alpha=alpha,
            )
        axs[i].set_title(dataset.replace("_", " ").title())

    plt.suptitle(f"{model} embeddings")
    # plt.legend()
    plt.tight_layout()
    model_folder = os.path.join(image_folder, model)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    plt.savefig(os.path.join(model_folder, f"{filename}.png"), dpi=300)
    plt.close()


def plot_labels_embeddings(
    data,
    labels,
    idxs,
    datasets,
    all_labels,
    model,
    image_folder,
    size: int = 8,
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
        model_folder = os.path.join(image_folder, model)
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        filename = f"{lab}.png".replace("/", "_")
        plt.savefig(os.path.join(model_folder, filename), dpi=300)
        plt.close()


DATA_FOLDER = "data/processed_embeddings"

IMAGE_FOLDER = "figures/2d_plots"
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

loaded_embeddings, labels, all_labels = load_joint_processed_data(DATA_FOLDER)
# loaded embeddings {(model, dataset): np.array of shape (n_file*n_frame, emb_dim)}
# labels {(model, dataset): list (n_file*n_frame) of list of labels}

models = sorted(set([m for m, _ in loaded_embeddings.keys()]))
datasets = sorted(set([d for _, d in loaded_embeddings.keys()]))

for model in models:
    print(f"Plotting for model {model}")
    image_folder = os.path.join(IMAGE_FOLDER, model)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # we want a single array for every emb of the model (i.e., for each dataset)
    # and a list of list of labels, w same length as emb.shape[0].
    # Also, to do the plots, we need a dict {dataset: (start_idx, end_idx)}.
    # We will compute it while concatenating the embeddings
    idxs = {}
    start, end = 0, 0
    model_emb = []
    for dataset in datasets:
        e = loaded_embeddings[(model, dataset)]
        end += e.shape[0]
        idxs[dataset] = (start, end)
        start = end

        model_emb.append(e)
    model_emb = np.concatenate(model_emb)

    model_labels = []
    for dataset in datasets:
        model_labels.extend(labels[(model, dataset)])

    pca = PCA(n_components=50)
    pca_data = pca.fit_transform(model_emb)
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

    # we can plot
    plot_overall_embeddings(
        tsne_data, model_labels, idxs, datasets, all_labels, "tsne", image_folder
    )
    # plot_overall_embeddings(umap_data, model_labels, idxs, datasets, all_labels, "umap")

    # plot first level of the taxonomy
    first_levels = tuple(set([l.split("/")[0] for l in all_labels]))
    plot_overall_embeddings(
        tsne_data,
        model_labels,
        idxs,
        datasets,
        first_levels,
        "tsne",
        image_folder,
        size=8,
        filename="all_datasets_first",
    )
    # plot_overall_embeddings(umap_data, model_labels, idxs, datasets, first_levels, "umap")

    plot_labels_embeddings(
        tsne_data, model_labels, idxs, datasets, all_labels, "tsne", image_folder
    )
    # plot_labels_embeddings(umap_data, model_labels, idxs, datasets, all_labels, "umap")

"""
# datasets = ["sons_al_balco_2020", "sons_al_balco_2021", "sabadell", "granollers"]
datasets = ["sons_al_balco_2020", "sons_al_balco_2021"]
models = ["perch", "birdnet", "yamnet"]
for model in models:
    print(f"Plotting for model {model}")
    image_folder = os.path.join(IMAGE_FOLDER, model)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    embeddings = {}
    for dataset in datasets:
        # load embeddings and labels with pickle
        with open(
            os.path.join(DATA_FOLDER, model, dataset, "embeddings.pkl"), "rb"
        ) as f:
            emb = pickle.load(f)
        with open(os.path.join(DATA_FOLDER, model, dataset, "labels.pkl"), "rb") as f:
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

    # plot first level of the taxonomy
    first_levels = tuple(set([l.split("/")[0] for l in all_labels]))
    plot_overall_embeddings(
        tsne_data, labels, idxs, datasets, first_levels, "tsne", image_folder
    )
    # plot_overall_embeddings(umap_data, labels, idxs, datasets, first_levels, "umap")

    plot_labels_embeddings(
        tsne_data, labels, idxs, datasets, all_labels, "tsne", image_folder
    )
    # plot_labels_embeddings(umap_data, labels, idxs, datasets, all_labels, "umap")
"""
