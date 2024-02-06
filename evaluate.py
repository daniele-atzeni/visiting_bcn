import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    multilabel_confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)


from utils import get_taxonomy_leaves

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
taxonomy_names = get_taxonomy_leaves()


def plot_history(history):
    metrics = ["loss"]
    for metric in metrics:
        name = metric.replace("_", " ").capitalize()
        plt.figure(figsize=(8, 6))
        plt.plot(history.epoch, history.history[metric], color=colors[0], label="Train")
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            color=colors[0],
            linestyle="--",
            label="Val",
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1]])
        else:
            plt.ylim([0, 1])

        plt.legend()


def plot_cm(true_y, pred_y, threshold=0.5):
    pred_labels = pred_y > threshold
    for i, cm in enumerate(multilabel_confusion_matrix(true_y, pred_labels)):
        # check if cm has something other than true negatives
        if cm[0, 0] == cm.ravel().sum():
            continue
        plt.figure(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(
            "Confusion matrix for {} @{:.2f}".format(taxonomy_names[i], threshold)
        )
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")

        plt.show()


def compute_metrics(test_y, pred_y, threshold=0.5):
    pred_y = pred_y > threshold

    acc = accuracy_score(test_y, pred_y)
    bacc = [
        balanced_accuracy_score(test_y[:, i], pred_y[:, i])
        for i in range(pred_y.shape[1])
    ]
    avg_bacc = sum(bacc) / len(bacc)
    prec_macro = precision_score(test_y, pred_y, average="macro", zero_division=0)
    rec_macro = recall_score(test_y, pred_y, average="macro", zero_division=0)
    f1_macro = f1_score(test_y, pred_y, average="macro", zero_division=0)
    prec_micro = precision_score(test_y, pred_y, average="micro", zero_division=0)
    rec_micro = recall_score(test_y, pred_y, average="micro", zero_division=0)
    f1_micro = f1_score(test_y, pred_y, average="micro", zero_division=0)

    res = {
        "accuracy": acc,
        "balanced_accuracy": avg_bacc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_micro": prec_micro,
        "recall_micro": rec_micro,
        "f1_micro": f1_micro,
    }

    return res


def compute_per_class_metrics(test_y, pred_y, threshold=0.5):
    pred_y = pred_y > threshold

    per_class_metrics = {}
    for i, label in enumerate(taxonomy_names):
        per_class_metrics[label] = {
            "precision": precision_score(test_y[:, i], pred_y[:, i], zero_division=0),
            "recall": recall_score(test_y[:, i], pred_y[:, i], zero_division=0),
            "f1": f1_score(test_y[:, i], pred_y[:, i], zero_division=0),
        }

    return per_class_metrics


def evaluate(y_true, y_pred, threshold=0.5, plot_confusion_matrices=True):
    """Evaluate a model on a test set given the true labels and the predicted labels
    and plot some results

    Args:
        y_true (np.array): True labels.
        y_pred (np.array): Predicted labels.

    Returns:
        dict: Dictionary containing the evaluation metrics.
    """
    if plot_confusion_matrices:
        plot_cm(y_true, y_pred, threshold=threshold)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, threshold=threshold)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"\t{k}: {v:.4f}")

    # Compute per-class metrics
    per_class_metrics = compute_per_class_metrics(y_true, y_pred, threshold=threshold)
    print("Per-class metrics:")
    for label, metrics in per_class_metrics.items():
        print(f"\t{label}:")
        for k, v in metrics.items():
            print(f"\t\t{k}: {v:.4f}")

    return
