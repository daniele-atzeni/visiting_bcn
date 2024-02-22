import os
import json
import pickle

from utils_new import (
    load_embeddings,
    modify_embedding_windows,
    create_yamnet_frame_labels,
    modify_yamnet_labels_windows,
)


AGGREGATION = "mean"
FINAL_WINDOW_SIZE = 3  # seconds

main_folder = "data/embeddings"
result_folder = "data/embeddings_consistent"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

labels_sab_2020_yamnet = create_yamnet_frame_labels(
    "data/sons_al_balco_labels/SAB-AudioTagging-2020.json"
)
labels_sab_2021_yamnet = create_yamnet_frame_labels(
    "data/sons_al_balco_labels/SAB-AudioTagging-2021.json"
)
new_labels_2020 = []
print("Modifying 2020 labels...")
for name, lab_dict in sorted(labels_sab_2020_yamnet.items()):
    new_el = modify_yamnet_labels_windows(lab_dict, FINAL_WINDOW_SIZE)
    new_labels_2020.append(new_el)
new_labels_2021 = []
print("Modifying 2021 labels...")
for name, lab_dict in sorted(labels_sab_2021_yamnet.items()):
    new_el = modify_yamnet_labels_windows(lab_dict, FINAL_WINDOW_SIZE)
    new_labels_2021.append(new_el)
print("Saving labels...")
with open(os.path.join(result_folder, "labels_2020.pkl"), "wb") as f:
    pickle.dump(new_labels_2020, f)
with open(os.path.join(result_folder, "labels_2021.pkl"), "wb") as f:
    pickle.dump(new_labels_2021, f)
print("Done")

for model in os.listdir(main_folder):
    orig_model_folder = os.path.join(main_folder, model)
    res_model_folder = os.path.join(result_folder, model)
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

        # let's load the list of the embeddings
        print(f"Modifying {model} {dataset} embeddings...")
        orig_embeddings = load_embeddings(orig_dataset_folder, model)
        # make embeddings consistent
        new_embeddings = modify_embedding_windows(
            orig_embeddings,
            config["window_size"],
            config["window_shift"],
            FINAL_WINDOW_SIZE,
            AGGREGATION,
        )
        # save the results with pickle
        print("Saving embeddings...")
        with open(os.path.join(res_dataset_folder, "embeddings.pkl"), "wb") as f:
            pickle.dump(new_embeddings, f)
        print("Done")
