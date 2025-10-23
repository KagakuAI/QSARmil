import joblib
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
from tqdm import tqdm

from qsarmil.utils.logging import FailedConformer, FailedMolecule

RDLogger.DisableLog("rdApp.*")


class FragmentGenerator:
    """Generate molecular fragments using RDKit BRICS decomposition.

    Converts molecules into sets of fragments, with optional parallelization
    and progress tracking.

    Args:
        num_cpu (int): Number of CPU threads for parallel processing.
        verbose (bool): Whether to display a progress bar.
    """

    def __init__(self, num_cpu=1, verbose=True):
        """Initialize the FragmentGenerator.

        Args:
            num_cpu (int): Number of CPU threads.
            verbose (bool): Whether to show progress bar.
        """
        super().__init__()

        self.num_cpu = num_cpu
        self.verbose = verbose

    def _generate_fragments(self, mol):
        """Generate fragments for a single molecule using BRICS decomposition.

        Args:
            mol (rdkit.Chem.Mol or FailedMolecule/FailedConformer): Input molecule.

        Returns:
            list[rdkit.Chem.Mol] or FailedMolecule: List of fragment molecules,
            or a FailedMolecule if fragmentation failed.
        """
        if isinstance(mol, (FailedMolecule, FailedConformer)):
            return mol
        try:
            frag_smiles_set = BRICS.BRICSDecompose(mol)
            frags = [Chem.MolFromSmiles(smi) for smi in frag_smiles_set if smi]
            frags = [f for f in frags if f is not None]
        except Exception:
            return FailedMolecule(mol)

        return frags

    def run(self, list_of_mols):
        """Generate fragments for a list of molecules in parallel with progress
        tracking.

        Args:
            list_of_mols (list): List of RDKit molecules to fragment.

        Returns:
            list: List of fragment lists or FailedMolecule objects for each input molecule.
        """
        with tqdm(total=len(list_of_mols), desc="Generating fragments", disable=not self.verbose) as progress_bar:

            class TqdmCallback(joblib.parallel.BatchCompletionCallBack):
                def __call__(self, *args, **kwargs):
                    progress_bar.update(self.batch_size)
                    return super().__call__(*args, **kwargs)

            # Patch joblib to use our callback
            old_callback = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = TqdmCallback

            try:
                results = Parallel(n_jobs=self.num_cpu, backend="threading")(
                    delayed(self._generate_fragments)(mol) for mol in list_of_mols
                )
            finally:
                joblib.parallel.BatchCompletionCallBack = old_callback  # Restore

        return results
