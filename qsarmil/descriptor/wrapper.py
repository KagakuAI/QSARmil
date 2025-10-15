import joblib
from tqdm import tqdm
import numpy as np
from qsarmil.utils.logging import FailedDescriptor
from joblib import Parallel, delayed


class DescriptorWrapper:
    def __init__(self, transformer, num_cpu=2, verbose=True):
        super().__init__()
        self.transformer = transformer
        self.num_cpu = num_cpu
        self.verbose = verbose

    def _ce2bag(self, mol):
        bag = []
        for conf in mol.GetConformers():
            x = self.transformer(mol, conformer_id=conf.GetId())
            bag.append(x)

        return np.array(bag)

    def _transform(self, mol):
        try:
            x = self._ce2bag(mol)
        except Exception as e:
            print(e)
            x = FailedDescriptor(mol)
        return x

    def run(self, list_of_mols):
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





