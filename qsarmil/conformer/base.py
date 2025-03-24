from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from joblib import Parallel, delayed
from qsarmil.utils.logging import FailedMolecule, FailedConformer

RDLogger.DisableLog('rdApp.*')


class ConformerGenerator:
    def __init__(self, num_conf=10, e_thresh=None, num_cpu=1):
        super().__init__()

        self.num_conf = num_conf
        self.e_thresh = e_thresh
        self.num_cpu = num_cpu

    def _prepare_molecule(self, mol):
        pass

    def _embedd_conformers(self, mol):
        pass

    def _optimize_conformers(self, mol):
        for conf in mol.GetConformers():
            AllChem.UFFOptimizeMolecule(mol, confId=conf.GetId())
        return mol

    def _generate_conformers(self, mol):
        if isinstance(mol, (FailedMolecule, FailedConformer)):
            return mol
        try:
            mol = self._embedd_conformers(mol)
            if not mol.GetNumConformers():
                return FailedConformer(mol)
            mol = self._optimize_conformers(mol)
        except:
            return FailedConformer(mol)

        if self.e_thresh:
            mol = filter_by_energy(mol, self.e_thresh)

        return mol

    def generate(self, list_of_mols):
        futures = Parallel(n_jobs=self.num_cpu)(delayed(self._generate_conformers)(mol) for mol in list_of_mols)
        return [mol for mol in futures]



def filter_by_energy(mol, e_thresh=1):
    conf_energy_list = []
    for conf in mol.GetConformers():
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
        if ff is None:
            continue
        conf_energy_list.append((conf.GetId(), ff.CalcEnergy()))
    conf_energy_list = sorted(conf_energy_list, key=lambda x: x[1])

    # filter conformers
    min_energy = conf_energy_list[0][1]
    for conf_id, conf_energy in conf_energy_list[1:]:
        if conf_energy - min_energy >= e_thresh:
            mol.RemoveConformer(conf_id)

    return mol


def filter_by_rmsd(mol, rmsd_thresh=2):
    pass
