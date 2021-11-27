import numpy as np
import pandas as pd
from typing import List, Tuple

from data.encode import MolGraph, get_graph_from_smiles


def load_data(data_name, set_name='train', max_num=-1) -> Tuple[List[MolGraph], np.ndarray]:
    if set_name == 'train':
        file_name = f'data/csv/{data_name}-train.csv'
    elif set_name == 'test':
        file_name = f'data/csv/{data_name}-test.csv'
    elif set_name == 'eval':
        file_name = f'data/csv/{data_name}-eval.csv'
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
