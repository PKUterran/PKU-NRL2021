import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def check_answer(answer, target, metric='rmse'):
    pass


check_answer('answer/ESOL-answer.csv', 'data/csv/ESOL-eval.csv')
check_answer('answer/Lipop-.csv', 'data/csv/Lipop-eval.csv')
check_answer('answer/sars-aanswernswer.csv', 'data/csv/sars-eval.csv', metric='roc-auc')
