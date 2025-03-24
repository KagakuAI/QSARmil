import numpy as np
from miprop.utils.logging import FailedDescriptor


class DescriptorWrapper:
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def _mol2bag(self, mol):
        bag = []
        for conf in mol.GetConformers():
            x = self.transformer(mol, conformer_id=conf.GetId())
            bag.append(x)

        return np.array(bag)

    def transform(self, list_of_mols):
        list_of_bags = []
        for mol_id, mol in enumerate(list_of_mols):

            try:
                x = self._mol2bag(mol)
            except Exception as e:
                x = FailedDescriptor(mol)
            list_of_bags.append(x)

        return list_of_bags
