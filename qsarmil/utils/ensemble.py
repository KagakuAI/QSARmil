from typing import Iterable
from rdkit import Chem
from rdkit.Chem import Mol

from qsarmil.utils.logging import FailedConformer


class ConformerEnsemble(list[Mol]):
    """An ensemble of RDKit Mol objects, one per conformer."""

    def __init__(self, mol: Mol):
        super().__init__()

        if isinstance(mol, FailedConformer) or mol.GetNumConformers() == 0:
            raise ValueError("Input molecule has no conformers")

        for conf in mol.GetConformers():
            conf_mol = Chem.Mol(mol)
            conf_mol.RemoveAllConformers()
            conf_mol.AddConformer(conf, assignId=True)
            self.append(conf_mol)


class FragmentEnsemble(list[Mol]):
    """A list of RDKit Mol objects representing fragments."""
    def __init__(self, mols: Iterable[Mol] = ()):
        super().__init__(mols)


class MixtureEnsemble(list[Mol]):
    """A list of RDKit Mol objects representing compound mixtures."""
    def __init__(self, mols: Iterable[Mol] = ()):
        super().__init__(mols)