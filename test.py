from data.load_data import load_data

a, b = load_data('sars', set_name='test', max_num=10)
print(a[1].atom_features)
print(a[1].bond_features)
print(a[1].start_indices)
print(a[1].end_indices)
print(b[1])
