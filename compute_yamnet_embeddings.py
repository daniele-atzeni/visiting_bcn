import os

import librosa
import numpy as np

from yamnet import yamnet_model

from utils_new import create_audio_data_from_videos


def compute_embeddings(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # sons al balco 2020
    # load
    data_soa_2020 = []
    data_dir = "data/sons_al_balco_audios/audios_2020"
    for el in sorted(os.listdir(data_dir)):
        data_soa_2020.append(librosa.load(os.path.join(data_dir, el)))
    # compute embeddings
    yamnet_emb_soa_2020 = []
    for el, _ in data_soa_2020:  # librosa loads numpy and sampling rate
        _, emb, _ = yamnet_model(el)
        yamnet_emb_soa_2020.append(emb)
    # padding to the longest shape
    max_length = max([el.shape[0] for el in yamnet_emb_soa_2020])
    yamnet_emb_soa_2020 = [
        (
            el
            if el.shape[0] == max_length
            else np.concatenate((el, np.zeros((max_length - el.shape[0], el.shape[1]))))
        )
        for el in yamnet_emb_soa_2020
    ]
    # save
    yamnet_emb_soa_2020 = np.stack(yamnet_emb_soa_2020)
    np.save(
        os.path.join(
            folder, "yamnet", "sons_al_balco_2020", "sons_al_balco_2020_yamnet.npy"
        ),
        yamnet_emb_soa_2020,
    )
    # delete data
    del data_soa_2020
    del yamnet_emb_soa_2020

    # sons al balco 2021
    # load
    data_soa_2021 = []
    data_dir = "data/sons_al_balco_audios/audios_2021"
    for el in sorted(os.listdir(data_dir)):
        data_soa_2021.append(librosa.load(os.path.join(data_dir, el)))
    # compute embeddings
    yamnet_emb_soa_2021 = []
    for el, _ in data_soa_2021:  # librosa loads numpy and sampling rate
        _, emb, _ = yamnet_model(el)
        yamnet_emb_soa_2021.append(emb)
    # padding
    max_length = max([el.shape[0] for el in yamnet_emb_soa_2021])
    yamnet_emb_soa_2021 = [
        (
            el
            if el.shape[0] == max_length
            else np.concatenate((el, np.zeros((max_length - el.shape[0], el.shape[1]))))
        )
        for el in yamnet_emb_soa_2021
    ]
    # save
    yamnet_emb_soa_2021 = np.stack(yamnet_emb_soa_2021)
    np.save(
        os.path.join(
            folder, "yamnet", "sons_al_balco_2021", "sons_al_balco_2021_yamnet.npy"
        ),
        yamnet_emb_soa_2021,
    )
    # delete data
    del data_soa_2021
    del yamnet_emb_soa_2021

    # granollers
    # load
    filename = "data/Granollers.xlsx"
    pre_url = "https://enquestes.salle.url.edu/videos_sons/granollers/"
    data_granollers = create_audio_data_from_videos(filename, pre_url)
    # compute embeddings
    yamnet_emb_granollers = []
    for el, _ in data_granollers:
        _, emb, _ = yamnet_model(el)
        yamnet_emb_granollers.append(emb)
    # padding
    max_length = max([el.shape[0] for el in yamnet_emb_granollers])
    yamnet_emb_granollers = [
        (
            el
            if el.shape[0] == max_length
            else np.concatenate((el, np.zeros((max_length - el.shape[0], el.shape[1]))))
        )
        for el in yamnet_emb_granollers
    ]
    # save
    yamnet_emb_granollers = np.stack(yamnet_emb_granollers)
    np.save(
        os.path.join(folder, "granollers", "granollers_yamnet.npy"),
        yamnet_emb_granollers,
    )
    # delete data
    del data_granollers
    del yamnet_emb_granollers

    # sabadell
    # load
    filename = "data/Sabadell.xlsx"
    pre_url = "https://enquestes.salle.url.edu/videos_sons/sabadell/"
    data_sabadell = create_audio_data_from_videos(filename, pre_url)
    # compute embeddings
    yamnet_emb_sabadell = []
    for el, _ in data_sabadell:
        _, emb, _ = yamnet_model(el)
        yamnet_emb_sabadell.append(emb)
    # padding
    max_length = max([el.shape[0] for el in yamnet_emb_sabadell])
    yamnet_emb_sabadell = [
        (
            el
            if el.shape[0] == max_length
            else np.concatenate((el, np.zeros((max_length - el.shape[0], el.shape[1]))))
        )
        for el in yamnet_emb_sabadell
    ]
    # save
    yamnet_emb_sabadell = np.stack(yamnet_emb_sabadell)
    np.save(
        os.path.join(folder, "sabadell", "sabadell_yamnet.npy"), yamnet_emb_sabadell
    )


if __name__ == "__main__":
    compute_embeddings(folder="data/embeddings/")
