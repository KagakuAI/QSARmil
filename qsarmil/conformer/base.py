import joblib
from joblib import Parallel, delayed
from rdkit import RDLogger
from rdkit.Chem import AllChem, rdMolAlign
from tqdm import tqdm

from qsarmil.utils.logging import FailedConformer, FailedMolecule

RDLogger.DisableLog("rdApp.*")


class ConformerGenerator:
    """Generate and optimize molecular conformers with optional filtering.

    Args:
        num_conf (int): Number of conformers to generate per molecule.
        e_thresh (float, optional): Energy threshold for filtering high-energy conformers.
        rmsd_thresh (float, optional): RMSD threshold for filtering similar conformers.
        num_cpu (int): Number of CPU threads to use for parallel processing.
        verbose (bool): Whether to display a progress bar during generation.
    """

    def __init__(self, num_conf=10, e_thresh=None, rmsd_thresh=None, num_cpu=1, verbose=True):
        super().__init__()

        self.num_conf = num_conf
        self.e_thresh = e_thresh
        self.rmsd_thresh = rmsd_thresh
        self.num_cpu = num_cpu
        self.verbose = verbose

    def _prepare_molecule(self, mol):
        """Prepare the input molecule for conformer generation.

        This method should be implemented in a subclass to perform
        molecule-specific preprocessing, e.g., sanitization or protonation.

        Args:
            mol (rdkit.Chem.Mol): Input molecule.

        Returns:
            rdkit.Chem.Mol: Prepared molecule.
        """
        return NotImplemented

    def _embedd_conformers(self, mol):
        """Generate multiple 3D conformers for a molecule.

        Uses RDKit's ETKDGv3 method to embed conformers.

        Args:
            mol (rdkit.Chem.Mol): Molecule to embed conformers for.

        Returns:
            rdkit.Chem.Mol: Molecule with embedded conformers.
        """
        mol = self._prepare_molecule(mol)
        params = AllChem.ETKDGv3()
        params.numThreads = 0
        params.maxAttempts = 1000
        params.pruneRmsThresh = 0.1
        AllChem.EmbedMultipleConfs(mol, numConfs=self.num_conf, params=params)
        return mol

    def _optimize_conformers(self, mol):
        """Optimize all conformers of a molecule using UFF force field.

        Args:
            mol (rdkit.Chem.Mol): Molecule with conformers.

        Returns:
            rdkit.Chem.Mol: Molecule with optimized conformers.
        """
        for conf in mol.GetConformers():
            AllChem.UFFOptimizeMolecule(mol, confId=conf.GetId())
        return mol

    def _generate_conformers(self, mol):
        """Generate and optionally filter conformers for a molecule.

        Args:
            mol (rdkit.Chem.Mol or FailedMolecule/FailedConformer): Input molecule.

        Returns:
            rdkit.Chem.Mol or FailedConformer: Molecule with filtered conformers,
            or a FailedConformer if generation failed.
        """
        if isinstance(mol, (FailedMolecule, FailedConformer)):
            return mol
        try:
            mol = self._embedd_conformers(mol)
            if not mol.GetNumConformers():
                return FailedConformer(mol)
            mol = self._optimize_conformers(mol)
        except Exception:
            return FailedConformer(mol)

        if self.e_thresh is not None:
            mol = filter_by_energy(mol, self.e_thresh)

        if self.rmsd_thresh is not None:
            mol = filter_by_rmsd(mol, self.rmsd_thresh)

        return mol

    def run(self, list_of_mols):
        """Generate conformers for a list of molecules in parallel.

        Args:
            list_of_mols (list): List of RDKit molecules to process.

        Returns:
            list: List of molecules with generated conformers or FailedConformer objects.
        """
        with tqdm(total=len(list_of_mols), desc="Generating conformers", disable=not self.verbose) as progress_bar:

            class TqdmCallback(joblib.parallel.BatchCompletionCallBack):
                def __call__(self, *args, **kwargs):
                    progress_bar.update(self.batch_size)
                    return super().__call__(*args, **kwargs)

            # Patch joblib to use our callback
            old_callback = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = TqdmCallback

            try:
                results = Parallel(n_jobs=self.num_cpu, backend="threading")(
                    delayed(self._generate_conformers)(mol) for mol in list_of_mols
                )
            finally:
                joblib.parallel.BatchCompletionCallBack = old_callback

        return results


def filter_by_energy(mol, e_thresh=1):
    """Filter conformers of a molecule based on relative energy.

    Args:
        mol (rdkit.Chem.Mol): Molecule with conformers.
        e_thresh (float): Maximum allowed energy difference from the lowest-energy conformer.

    Returns:
        rdkit.Chem.Mol: Molecule with high-energy conformers removed.
    """
    conf_energy_list = []
    for conf in mol.GetConformers():
        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
        if ff is None:
            continue
        conf_energy_list.append((conf.GetId(), ff.CalcEnergy()))
    conf_energy_list = sorted(conf_energy_list, key=lambda x: x[1])

    min_energy = conf_energy_list[0][1]
    for conf_id, conf_energy in conf_energy_list[1:]:
        if conf_energy - min_energy >= e_thresh:
            mol.RemoveConformer(conf_id)

    return mol


def filter_by_rmsd(mol, rmsd_thresh=2):
    """Filter conformers of a molecule based on RMSD similarity.

    Args:
        mol (rdkit.Chem.Mol): Molecule with conformers.
        rmsd_thresh (float): Minimum RMSD between conformers to retain both.

    Returns:
        rdkit.Chem.Mol: Molecule with similar conformers removed.
    """
    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    to_remove = set()

    for i, conf_id_i in enumerate(conf_ids):
        if conf_id_i in to_remove:
            continue
        for conf_id_j in conf_ids[i + 1 :]:
            if conf_id_j in to_remove:
                continue
            rmsd = rdMolAlign.GetConformerRMS(mol, conf_id_i, conf_id_j, prealigned=False)
            if rmsd < rmsd_thresh:
                to_remove.add(conf_id_j)

    for conf_id in to_remove:
        mol.RemoveConformer(conf_id)

    return mol
