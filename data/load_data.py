import csv
import numpy as np
import pandas as pd
from typing import List, Tuple

from data.encode import MolGraph, get_graph_from_smiles


def load_data(data_name, set_name='train', max_num=-1) -> Tuple[List[MolGraph], np.ndarray]:
    if set_name == 'train':
        file_name = f'data/csvs/{data_name}-train.csv'
    elif set_name == 'test':
        file_name = f'data/csvs/{data_name}-test.csv'
    elif set_name == 'eval':
        file_name = f'data/csvs/{data_name}-eval.csv'
    else:
        assert False

    df = pd.read_csv(file_name)
    values: np.ndarray = df.values
    list_smiles = values[:, 0].astype(np.str)
    properties = values[:, 1:].astype(np.float)
    graphs = [get_graph_from_smiles(smiles) for smiles in list_smiles]

    if max_num != -1 and max_num < len(list_smiles):
        return graphs[:max_num], properties[:max_num, :]
    return graphs, properties


def output_answer(data_name, properties: np.ndarray):
    file_name = f'data/csvs/{data_name}-test.csv'
    df = pd.read_csv(file_name)
    values: np.ndarray = df.values
    list_smiles = values[:, 0].astype(np.str)
    with open(f'answer/{data_name}-pred.csv', 'w+', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['smiles'] + [f'target_{i}' for i in range(properties.shape[1])])
        for s, p in zip(list_smiles, properties):
            writer.writerow([s] + list(p))
