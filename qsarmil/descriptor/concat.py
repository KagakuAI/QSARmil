import joblib
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from qsarmil.utils.logging import FailedDescriptor


class DescriptorConcat:
    """Concatenate different descriptors"""

    def __init__(self, transformer_list, num_cpu=2, verbose=True):

        super().__init__()
        self.transformer_list = transformer_list
        self.num_cpu = num_cpu
        self.verbose = verbose

    def _ce2bag(self, mol):

        bag_concat = []
        for transformer in self.transformer_list:
            bag = []
            for conf in mol.GetConformers():
                bag = transformer(mol, conformer_id=conf.GetId())
            bag_concat.append(bag)
        bag_concat = np.hstack(bag_concat)
        return np.array(bag_concat)

    def _transform(self, mol):
        """Safely compute descriptors for a single molecule.

        Catches exceptions during descriptor calculation and returns a
        FailedDescriptor object if computation fails.

        Args:
            mol (rdkit.Chem.Mol): Input molecule.

        Returns:
            np.ndarray or FailedDescriptor: Descriptor array or failure object.
        """
        try:
            x = self._ce2bag(mol)
        except Exception as e:
            print(e)
            x = FailedDescriptor(mol)
        return x

    def run(self, list_of_mols):
        """Compute descriptors for a list of molecules in parallel with
        progress tracking.

        Args:
            list_of_mols (list): List of RDKit molecules to compute descriptors for.

        Returns:
            list: List of descriptor arrays or FailedDescriptor objects.
        """
        with tqdm(total=len(list_of_mols), desc="Calculating descriptors", disable=not self.verbose) as progress_bar:

            class TqdmCallback(joblib.parallel.BatchCompletionCallBack):
                def __call__(self, *args, **kwargs):
                    progress_bar.update(self.batch_size)
                    return super().__call__(*args, **kwargs)

            # Patch joblib to use our callback
            old_callback = joblib.parallel.BatchCompletionCallBack
            joblib.parallel.BatchCompletionCallBack = TqdmCallback

            try:
                results = Parallel(n_jobs=self.num_cpu, backend="threading")(
                    delayed(self._transform)(mol) for mol in list_of_mols
                )
            finally:
                joblib.parallel.BatchCompletionCallBack = old_callback

        return results
