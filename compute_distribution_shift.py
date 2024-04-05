import os
import pickle

import numpy as np
from scipy.stats import ks_2samp

from utils import load_embeddings, load_labels, compute_all_labels, aggregate_embeddings, aggregate_labels

DATA_FOLDER: str = "data"
EMB_FOLDER: str = "data/embeddings"
RESULTS_FOLDER: str = "results"

embeddings = load_embeddings(EMB_FOLDER)
labels = load_labels(DATA_FOLDER)
all_labels = compute_all_labels(labels)

models = embeddings.keys()
dataset_names = labels.keys()

# let's modify embeddings to be {model: {dataset: np.array (total_frames, features)}}
# and labels to be {dataset: [list (tot n frames) of lists (labels in frame)]}
embeddings = aggregate_embeddings(embeddings)
labels = aggregate_labels(labels)

# compute the average number latent features computed by the model for which
# there is a shift in the distribution across different datasets for each model
res = {}
for model in models:
    res_model_folder = os.path.join(RESULTS_FOLDER, model)
    if not os.path.exists(res_model_folder):
        os.makedirs(res_model_folder)
    print(f"Computing distribution shift for model {model}...")
    print(f"and saving them into {res_model_folder}")

    res[model] = {}
    e1 = embeddings[model]["sons_al_balco_2020"]    # list of numpy arrays (tot_frames, features)
    e2 = embeddings[model]["sons_al_balco_2021"]    # list of numpy arrays (tot_frames, features)
    e3 = embeddings[model]["granollers"]    # list of numpy arrays (tot_frames, features)
    n_features = e1.shape[1]

    res = {}
    for l in all_labels:
        res[l] = {}

        idx_1 = [
            idx
            for idx, lab in enumerate(labels["sons_al_balco_2020"])
            if l in lab
        ]
        idx_2 = [
            idx
            for idx, lab in enumerate(labels["sons_al_balco_2021"])
            if l in lab
        ]
        idx_3 = [
            idx
            for idx, lab in enumerate(labels["granollers"])
            if l in lab
        ]
        e1_lab = e1[idx_1]
        e2_lab = e2[idx_2]
        e3_lab = e3[idx_3]
        res[l]["n_2020"] = e1_lab.shape[0]
        res[l]["n_2021"] = e2_lab.shape[0]
        res[l]["n_granollers"] = e3_lab.shape[0]

        for comp_data, e_lab in {"2021":e2_lab, "granollers":e3_lab}.items():
            if res[l]["n_2020"] == 0 or res[l][f"n_{comp_data}"] == 0:
                continue
            res[l][comp_data] = {}

            # p-values of KS test for each feature
            print(f"Computing p-values for {model} and {l}...")
            res[l][comp_data]["p_values"] = [
                ks_2samp(e1_lab[:, i], e_lab[:, i]).pvalue  # type:ignore
                for i in range(n_features)
            ]

            # compute centroids and distances in 2020 and 2021/Granollers
            centr_20 = np.mean(e1_lab, axis=0)
            centr_comp = np.mean(e_lab, axis=0)
            res[l][comp_data]["avg_2020"] = centr_20
            res[l][comp_data][f"avg_{comp_data}"] = centr_comp
            res[l][comp_data]["centr_dist"] = np.linalg.norm(centr_20 - centr_comp)
            avg_dist_20 = np.mean([np.linalg.norm(e - centr_20) for e in e1_lab])
            avg_dist_comp = np.mean([np.linalg.norm(e - centr_comp) for e in e_lab])
            res[l][comp_data]["avg_dist_2020"] = avg_dist_20
            res[l][comp_data][f"avg_dist_{comp_data}"] = avg_dist_comp
            dist_20_comp = np.mean([np.linalg.norm(e - centr_20) for e in e_lab])
            dist_comp_20 = np.mean([np.linalg.norm(e - centr_comp) for e in e1_lab])
            res[l][comp_data]["dist_20_21"] = dist_20_comp
            res[l][comp_data]["dist_21_20"] = dist_comp_20

    # save results with pickle
    with open(os.path.join(res_model_folder, "shifts_results.pkl"), "wb") as f:
        pickle.dump(res, f)
