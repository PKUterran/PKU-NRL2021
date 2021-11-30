import numpy as np
import torch

from data.load_data import load_data, output_answer
from data.encode import num_atom_features, num_bond_features
from model.GNN import ThisIsNotAGNNAtAll

MODEL_DICT_DIR = 'model/pt'

SEED = 0
USE_CUDA = False

config = {
    'HIDDEN_DIM': 128,
}

np.random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed(USE_CUDA)

print('\tLoading Data...')
_, train_properties = load_data('ESOL')
list_mol_graph, _ = load_data('ESOL', set_name='test')
n_mol = len(list_mol_graph)
list_torch_graph = [(
    torch.FloatTensor(mol_graph.atom_features),
    torch.FloatTensor(mol_graph.bond_features),
    mol_graph.start_indices,
    mol_graph.end_indices
) for mol_graph in list_mol_graph]
train_properties = torch.FloatTensor(train_properties)
mean = torch.mean(train_properties, dim=0, keepdim=True)
std = torch.std(train_properties, dim=0, keepdim=True)


def normalize_prop(p: torch.Tensor) -> torch.Tensor:
    return (p - mean) / std


def denormalize_prop(p: torch.Tensor) -> torch.Tensor:
    return p * std + mean


print('\tBuilding Model...')
model = ThisIsNotAGNNAtAll(
    atom_dim=num_atom_features(),
    bond_dim=num_bond_features(),
    output_dim=1,
    config=config,
    use_cuda=USE_CUDA
)
model.load_state_dict(torch.load(f'{MODEL_DICT_DIR}/ThisIsNotAGNNAtAll-ESOL.pkl', map_location=torch.device('cpu')))
if USE_CUDA:
    model.cuda()

model.eval()
list_pred = []
for i, (af, bf, us, vs) in enumerate(list_torch_graph):
    if USE_CUDA:
        af, bf = af.cuda(), bf.cuda()
    pred = model.forward(af, bf, us, vs)
    list_pred.append(pred.cpu().detach())

total_pred = torch.vstack(list_pred)
total_pred = denormalize_prop(total_pred)

output_answer('ESOL', total_pred.numpy())
