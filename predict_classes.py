import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from utils import get_taxonomy_leaves
from yamnet import yamnet_model


# Function to predict the class of an audio file
def predict_audio_class(model_name, audio, aggregation="mean"):
    """
    model must be in ["yamnet", "fine_tuned", "sklearn"]
    audio is from librosa.load
    return a list of (class_name, probability) with the top 5 predictions
    """
    # Make the prediction with Yamnet, then we use the prediction if the model
    # name is yamnet, otherwise we use the embeddings
    scores, embeddings, _ = yamnet_model(audio)  # type: ignore problems with the model

    if model_name == "yamnet":
        # we will use weak labels (i.e. only one or more labels for each audio file)
        # so we will aggregate the scores (and embeddings)
        if aggregation == "mean":
            scores = tf.reduce_mean(scores, axis=0)
        elif aggregation == "max":
            scores = tf.reduce_max(scores, axis=0)
        else:
            raise ValueError("Aggregation method not supported")

        class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")  # type: ignore not my code
        yamnet_classes = list(pd.read_csv(class_map_path)["display_name"])
        # get top 5 preds
        top_5_values, top_5_indices = tf.math.top_k(scores, k=5)
        return [
            (scores[i].numpy(), yamnet_classes[i])
            for i in top_5_indices.numpy().squeeze()
        ]

    # embeddings = tf.reduce_mean(embeddings, axis=0)
    classes = get_taxonomy_leaves()
    if model_name == "fine_tuned":
        fine_tuned_model_handle = "ml_models/fine_tuned_model.keras"
        model = tf.keras.models.load_model(fine_tuned_model_handle)
        assert model, "Failed to load the fine tuned model"
        scores = model(embeddings)
        if aggregation == "mean":
            scores = tf.reduce_mean(scores, axis=0)
        elif aggregation == "max":
            scores = tf.reduce_max(scores, axis=0)
        else:
            raise ValueError("Aggregation method not supported")
        # get top 5 preds
        top_5_values, top_5_indices = tf.math.top_k(scores, k=5)
        return [
            (scores[i].numpy(), classes[i]) for i in top_5_indices.numpy().squeeze()
        ]
    if model_name == "sklearn":
        sklearn_model_handle = "ml_models/multilabel_sklearn_model.pkl"
        with open(sklearn_model_handle, "rb") as f:
            model = pickle.load(f)
        assert model, "Failed to load the sklearn model"
        scores = model.predict_proba(embeddings).squeeze()
        if aggregation == "mean":
            scores = np.mean(scores, axis=0)
        elif aggregation == "max":
            scores = np.max(scores, axis=0)
        else:
            raise ValueError("Aggregation method not supported")
        # get top 5 preds
        top_5_indices = np.argpartition(scores, -5)[-5:]
        top_5_indices = top_5_indices[np.argsort(-scores[top_5_indices])]
        return [(scores[i], classes[i]) for i in top_5_indices]

    raise ValueError("Model not supported")
