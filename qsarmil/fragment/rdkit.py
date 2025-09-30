import joblib
from tqdm import tqdm
from rdkit import RDLogger
from joblib import Parallel, delayed
from qsarmil.utils.logging import FailedMolecule, FailedConformer
from rdkit import Chem
from rdkit.Chem import BRICS
RDLogger.DisableLog('rdApp.*')


class FragmentGenerator:
    def __init__(self, num_cpu=1, verbose=True):
        super().__init__()

        self.num_cpu = num_cpu
        self.verbose = verbose

    def _generate_fragments(self, mol):
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
        with tqdm(total=len(list_of_mols), desc="Generating fragments", disable=not self.verbose) as progress_bar:
            class TqdmCallback(joblib.parallel.BatchCompletionCallBack):
                def __call__(self, *args, **kwargs):
                    progress_bar.update(self.batch_size)
                    return super().__call__(*args, **kwargs)

            # Patch joblib to use our callback
            old_callback = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = TqdmCallback

            try:
                results = Parallel(n_jobs=self.num_cpu)(
                    delayed(self._generate_fragments)(mol) for mol in list_of_mols
                )
            finally:
                joblib.parallel.BatchCompletionCallBack = old_callback  # Restore

        return results
