import csv
import numpy as np
import pandas as pd
import rdkit.Chem as Chem

from data.config import ESOL_CSV_PATH, LIPOP_CSV_PATH, SARS_CSV_PATH


def generate_csv(name):
    n_mol = len(smiles)
    n_prop = properties.shape[1]
    assert n_mol == properties.shape[0]

    train_num = int(n_mol * 0.8)
    seq = np.random.permutation(n_mol)
    train_mask = seq[:train_num]
    test_mask = seq[train_num:]
    train_smiles, test_smiles = smiles[train_mask], smiles[test_mask]
    train_properties, test_properties = properties[train_mask, :], properties[test_mask, :]

    with open(f'data/{name}-train.csv', 'w+', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['smiles'] + [f'target_{i}' for i in range(n_prop)])
        for s, p in zip(train_smiles, train_properties):
            writer.writerow([s] + list(p))

    with open(f'data/{name}-test.csv', 'w+', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['smiles'])
        for s in test_smiles:
            writer.writerow([s])

    with open(f'data/{name}-eval.csv', 'w+', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['smiles'] + [f'target_{i}' for i in range(n_prop)])
        for s, p in zip(test_smiles, test_properties):
            writer.writerow([s] + list(p))


# ESOL
df = pd.read_csv(ESOL_CSV_PATH)
values: np.ndarray = df.values
smiles = values[:, 9].astype(np.str)
properties = values[:, 8: 9].astype(np.float)
generate_csv('ESOL')


# Lipop
df = pd.read_csv(LIPOP_CSV_PATH)
values: np.ndarray = df.values
smiles = values[:, 2].astype(np.str)
properties = values[:, 1: 2].astype(np.float)
generate_csv('Lipop')

# sars
df = pd.read_csv(SARS_CSV_PATH, dtype=np.str)
values: np.ndarray = df.values
smiles = values[:, 0]
properties = values[:, 1: 14]
mols = [Chem.MolFromSmiles(s) for s in smiles]
mask = [i for i, m in enumerate(mols) if m is not None]
smiles = np.array([smiles[i] for i in mask])
properties = properties[mask, :]
generate_csv('sars')
