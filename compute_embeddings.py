import os
import pickle

import numpy as np

from embeddings import compute_openl3, compute_wav2vec, compute_yamnet


def aggr_frames(emb:np.ndarray, n:int) -> np.ndarray:
    frame_to_aggr = emb.shape[0] // n
    frame_to_aggr = frame_to_aggr if emb.shape[0] % n == 0 else frame_to_aggr + 1
    new_embs = []
    for i in range(n):
        new_emb = np.mean(emb[i: i + frame_to_aggr], axis=0)
        new_embs.append(new_emb)
    return np.stack(new_embs, axis=0)


MODELS2FUNC = {
    "yamnet" : compute_yamnet, 
    "wav2vec" : compute_wav2vec, 
    "openl3": compute_openl3
    }


def compute_embeddings(data_folder:str, res_folder: str) -> None:
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    datasets = [el.split("_audio")[0] for el in os.listdir(data_folder) if "audio" in el]
    models = MODELS2FUNC.keys()

    for dataset in datasets:
        dataset_folder = os.path.join(data_folder, f"{dataset}_audios")
        for filename in os.listdir(dataset_folder):
            if filename.split('.')[-1] != "wav":
                continue
            filepath = os.path.join(dataset_folder, filename)
            embs = {model: MODELS2FUNC[model](filepath) for model in models}
            
            # combine frames in order to have the same numbers
            min_n_frame = min([e.shape[0] for e in embs.values()])
            new_embs = {model: aggr_frames(emb, min_n_frame) for model, emb in embs.items()}

            # save results as numpy arrays (n_frames, emb_dim)
            for model in models:
                emb_folder = os.path.join(res_folder, model, dataset)
                if not os.path.exists(emb_folder):
                    os.makedirs(emb_folder)
                with open(os.path.join(emb_folder, f"{filename}.pkl"), "wb") as f:
                    pickle.dump(new_embs[model], f)

if __name__ == "__main__":
    data_folder = "data"
    res_folder = "data/test_embeddings"
    compute_embeddings(data_folder, res_folder)
