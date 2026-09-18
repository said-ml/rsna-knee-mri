import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(predictions, targets, target_names):

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    aucs = {}

    for i, name in enumerate(target_names):

        y_true = targets[:, i]
        y_pred = predictions[:, i]

        # AUC is undefined if validation contains only one class.
        if np.unique(y_true).size < 2:
            aucs[name] = float("nan")
        else:
            aucs[name] = float(
                roc_auc_score(y_true, y_pred)
            )

    valid = [
        value
        for value in aucs.values()
        if np.isfinite(value)
    ]

    macro_auc = (
        float(np.mean(valid))
        if valid
        else float("nan")
    )

    return aucs, macro_auc
