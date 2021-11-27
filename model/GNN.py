import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any


class ThisIsNotAGNNAtAll(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, output_dim: int, config: Dict[str, Any], use_cuda=False):
        super(ThisIsNotAGNNAtAll, self).__init__()
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.output_dim = output_dim
        self.hidden_dim: int = config['HIDDEN_DIM']

        self.use_cuda = use_cuda
        self.atom_embed = nn.Linear(self.atom_dim, self.hidden_dim)
        self.bond_embed = nn.Linear(self.bond_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.output_embed = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, atom_features: torch.Tensor, bond_features: torch.Tensor,
                start_indices: np.ndarray, end_indices: np.ndarray) -> torch.Tensor:
        hidden_atom = torch.sum(self.relu(self.atom_embed(atom_features)), dim=0)
        hidden_bond = torch.sum(self.relu(self.bond_embed(bond_features)), dim=0)
        return self.output_embed(hidden_atom + hidden_bond)
