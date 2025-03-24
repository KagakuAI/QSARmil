from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class FailedMolecule:
    def __init__(self, smiles):
        super().__init__()
        self.smiles = smiles

    def __str__(self):
        return f'{self.smiles} -> SMILES parsing failed'


class FailedConformer:
    def __init__(self, mol):
        super().__init__()
        self.mol = mol

    def __str__(self):
        smi = Chem.MolToSmiles(self.mol)
        return f'{smi} -> conformer generation failed'


class FailedDescriptor:
    def __init__(self, mol):
        super().__init__()
        self.mol = mol

    def __str__(self):
        smi = Chem.MolToSmiles(self.mol)
        return f'{smi} -> descriptor calculation failed'
