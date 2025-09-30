from rdkit import Chem
from rdkit.Chem import AllChem
from qsarmil.conformer.base import ConformerGenerator


class RDKitConformerGenerator(ConformerGenerator):
    def __init__(self, num_conf=10, e_thresh=None, num_cpu=1, verbose=True):
        super().__init__(num_conf=num_conf, e_thresh=e_thresh, num_cpu=num_cpu, verbose=verbose)

    def _prepare_molecule(self, mol):
        mol = Chem.AddHs(mol)
        return mol

    def _embedd_conformers(self, mol):
        mol = self._prepare_molecule(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=self.num_conf, maxAttempts=700, randomSeed=42)
        return mol
