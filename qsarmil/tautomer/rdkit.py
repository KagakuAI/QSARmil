from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from qsarmil.tautomer.base import TautomerGenerator


class RDKitTautomerGenerator(TautomerGenerator):
    def __init__(self, num_taut=10):
        super().__init__(num_taut=num_taut)

    def _prepare_molecule(self, mol):
        return mol

    def _embedd_tautomers(self, mol):
        enumerator = rdMolStandardize.TautomerEnumerator()
        tautomers = enumerator.Enumerate(mol)
        return tautomers


