import os
import pickle

import numpy as np
from scipy.stats import ks_2samp

from utils import load_joint_processed_data


DATA_FOLDER: str = "data/processed_embeddings"
RESULTS_FOLDER: str = "results"

models = os.listdir(DATA_FOLDER)

# dataset_names = os.listdir(os.path.join(data_folder, models[0]))
dataset_names = ["sons_al_balco_2020", "sons_al_balco_2021"]

embeddings, labels, all_labels = load_joint_processed_data(DATA_FOLDER)

# compute the average number latent features computed by the model for which
# there is a shift in the distribution across different datasets for each model
res = {}
for model in models:
    res_model_folder = os.path.join(RESULTS_FOLDER, model)
    if not os.path.exists(res_model_folder):
        os.makedirs(res_model_folder)

    res[model] = {}
    e1 = embeddings[(model, "sons_al_balco_2020")]
    e2 = embeddings[(model, "sons_al_balco_2021")]
    n_features = e1.shape[1]

    res = {}
    for l in all_labels:
        res[l] = {}

        idx_1 = [
            idx
            for idx, lab in enumerate(labels[(model, "sons_al_balco_2020")])
            if l in lab
        ]
        idx_2 = [
            idx
            for idx, lab in enumerate(labels[(model, "sons_al_balco_2021")])
            if l in lab
        ]
        e1_lab = e1[idx_1]
        e2_lab = e2[idx_2]
        res[l]["n_2020"] = e1_lab.shape[0]
        res[l]["n_2021"] = e2_lab.shape[0]
        if res[l]["n_2020"] == 0 or res[l]["n_2021"] == 0:
            continue

        # p-values of KS test for each feature
        print(f"Computing p-values for {model} and {l}...")
        res[l]["p_values"] = [
            ks_2samp(e1_lab[:, i], e2_lab[:, i]).pvalue  # type:ignore
            for i in range(n_features)
        ]

        # compute centroids and distances in 2020 and 2021
        centr_20 = np.mean(e1_lab, axis=0)
        centr_21 = np.mean(e2_lab, axis=0)
        res[l]["avg_2020"] = centr_20
        res[l]["avg_2021"] = centr_21
        res[l]["centr_dist"] = np.linalg.norm(centr_20 - centr_21)
        avg_dist_20 = np.mean([np.linalg.norm(e - centr_20) for e in e1_lab])
        avg_dist_21 = np.mean([np.linalg.norm(e - centr_21) for e in e2_lab])
        res[l]["avg_dist_2020"] = avg_dist_20
        res[l]["avg_dist_2021"] = avg_dist_21
        dist_20_21 = np.mean([np.linalg.norm(e - centr_20) for e in e2_lab])
        dist_21_20 = np.mean([np.linalg.norm(e - centr_21) for e in e1_lab])
        res[l]["dist_20_21"] = dist_20_21
        res[l]["dist_21_20"] = dist_21_20

    # save results with pickle
    with open(os.path.join(res_model_folder, "shifts_results.pkl"), "wb") as f:
        pickle.dump(res, f)
