import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def check_answer(pred_path, target_path, metric='rmse'):
    pred = pd.read_csv(pred_path).values[:, 1:].astype(np.float)
    target = pd.read_csv(target_path).values[:, 1:].astype(np.float)
    assert pred.shape == target.shape

    if metric == 'rmse':
        rmse = ((pred - target) ** 2).sum(axis=-1).mean() ** 0.5
        print(f'RMSE: {rmse:.4f}')
    elif metric == 'roc-auc':
        pred, target = pred.astype(np.int), target.astype(np.int)
        list_roc = []
        for i in range(target.shape[1]):
            p, t = pred[:, i], target[:, i]
            not_nan_mask = np.logical_and(np.logical_not(np.isnan(t)),
                                          np.logical_and(np.greater(t, -0.001), np.greater(3.001, t)))
            p, t = p[not_nan_mask], t[not_nan_mask]
            p = np.eye(4)[p]
            t_mask = sorted(set(t))
            p = p[:, t_mask]
            p = p + (1 - np.sum(p, axis=-1, keepdims=True)) / len(t_mask)
            list_roc.append(roc_auc_score(t, p, multi_class='ovo'))
        roc = np.mean(list_roc)
        print(f'ROC-AUC: {roc:.4f}')


check_answer('answer/ESOL-pred.csv', 'data/csvs/ESOL-eval.csv')
check_answer('answer/Lipop-pred.csv', 'data/csvs/Lipop-eval.csv')
check_answer('answer/sars-pred.csv', 'data/csvs/sars-eval.csv', metric='roc-auc')
