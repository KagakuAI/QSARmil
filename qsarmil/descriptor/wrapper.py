import joblib
from tqdm import tqdm
import numpy as np
from qsarmil.utils.logging import FailedDescriptor
from joblib import Parallel, delayed


class DescriptorWrapper:
    """Wrapper to compute molecular descriptors for multiple conformers in parallel.

    Converts a molecule into a "bag" of descriptor vectors, one per conformer,
    with optional parallelization and progress tracking.

    Args:
        transformer (callable): Descriptor function or object that accepts a molecule
            and optional conformer ID, returning a descriptor vector.
        num_cpu (int): Number of CPU threads for parallel processing.
        verbose (bool): Whether to display a progress bar.
    """

    def __init__(self, transformer, num_cpu=2, verbose=True):
        """Initialize the descriptor wrapper.

        Args:
            transformer (callable): Descriptor function or object.
            num_cpu (int): Number of CPU threads.
            verbose (bool): Whether to show progress bar.
        """
        super().__init__()
        self.transformer = transformer
        self.num_cpu = num_cpu
        self.verbose = verbose

    def _ce2bag(self, mol):
        """Convert a molecule into a bag of descriptor vectors (one per conformer).

        Args:
            mol (rdkit.Chem.Mol): Molecule with one or more conformers.

        Returns:
            np.ndarray: Array of descriptor vectors, shape (num_conformers, descriptor_length).
        """
        bag = []
        for conf in mol.GetConformers():
            x = self.transformer(mol, conformer_id=conf.GetId())
            bag.append(x)

        return np.array(bag)

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
        """Compute descriptors for a list of molecules in parallel with progress tracking.

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
                results = Parallel(n_jobs=self.num_cpu, backend='threading')(
                    delayed(self._transform)(mol) for mol in list_of_mols
                )
            finally:
                joblib.parallel.BatchCompletionCallBack = old_callback

        return results
