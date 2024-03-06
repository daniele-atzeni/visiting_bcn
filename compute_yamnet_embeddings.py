import os
import pickle

import librosa
import numpy as np

from yamnet import yamnet_model

from utils import create_audio_data_from_videos


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
        _, emb, _ = yamnet_model(el)  # type:ignore yamnet is callable
        yamnet_emb_soa_2020.append(emb)
    # save with pickle
    dest_folder = os.path.join(folder, "yamnet", "sons_al_balco_2020")
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    with open(os.path.join(dest_folder, "sons_al_balco_2020_yamnet.pkl"), "wb") as f:
        pickle.dump(yamnet_emb_soa_2020, f)
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
        _, emb, _ = yamnet_model(el)  # type:ignore yamnet is callable
        yamnet_emb_soa_2021.append(emb)
    # save
    dest_folder = os.path.join(folder, "yamnet", "sons_al_balco_2021")
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    with open(os.path.join(dest_folder, "sons_al_balco_2021_yamnet.pkl"), "wb") as f:
        pickle.dump(yamnet_emb_soa_2021, f)
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
        _, emb, _ = yamnet_model(el)  # type:ignore yamnet is callable
        yamnet_emb_granollers.append(emb)

    # save
    dest_folder = os.path.join(folder, "yamnet", "granollers")
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    with open(os.path.join(dest_folder, "granollers_yamnet.npy"), "wb") as f:
        pickle.dump(yamnet_emb_granollers, f)
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
        _, emb, _ = yamnet_model(el)  # type:ignore yamnet is callable
        yamnet_emb_sabadell.append(emb)
    # save
    yamnet_emb_sabadell = np.stack(yamnet_emb_sabadell)
    dest_folder = os.path.join(folder, "yamnet", "sabadell")
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    dest_file = os.path.join(dest_folder, "sabadell_yamnet.npy")
    np.save(dest_file, yamnet_emb_sabadell)


if __name__ == "__main__":
    compute_embeddings(folder="data/embeddings/")
