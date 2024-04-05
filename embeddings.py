import os
import pickle

import librosa
import numpy as np

from utils import create_audio_data_from_videos

# yamnet
import tensorflow_hub as hub

# openl3
import openl3
import soundfile as sf

# wav2vec
from transformers import AutoProcessor, Wav2Vec2Model
import torch


def compute_openl3(filename:str) -> np.ndarray:
    audio, sr = sf.read(filename)
    emb, ts = openl3.get_audio_embedding(audio, 
                                     sr, 
                                     hop_size=0.5, 
                                     content_type="env", 
                                     embedding_size=512, 
                                     input_repr="mel256", 
                                     center=False, 
                                     frontend="librosa",
                                     )
    return emb


def compute_wav2vec(filename:str, sr=16000) -> np.ndarray:
    audio, orig_sr = librosa.load(filename)
    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    # audio file is decoded on the fly
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states.numpy().squeeze()

def compute_yamnet(filename:str) -> np.ndarray:
    audio, _ = librosa.load(filename)

    yamnet_model_handle = "https://tfhub.dev/google/yamnet/1"
    yamnet_model = hub.load(yamnet_model_handle)
    _, emb, _ = yamnet_model(audio)
    # take only even elements of emb, because of frame shift
    #idxs = [i for i in range(emb.shape[0]) if i%2==0]
    #emb = emb.numpy()[idxs]
    return emb.numpy()


if __name__ == "__main__":
    filename = '/Users/platypus/Desktop/visiting_bcn/data/sons_al_balco_audios/audios_2020/fu_xw5e8nskn2zyrra.wav'

    emb1 = compute_openl3(filename)
    print("openl3", emb1.shape, type(emb1))
    emb2 = compute_wav2vec(filename)
    print("wav2vec", emb2.shape, type(emb2))
    emb3 = compute_yamnet(filename)
    print("yamnet", emb3.shape, type(emb3))



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

