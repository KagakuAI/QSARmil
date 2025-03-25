from rdkit import Chem



class TautomerGenerator:
    def __init__(self, num_taut=10):
        super().__init__()

        self.num_taut = num_taut

    def _prepare_molecule(self, mol):
        pass

    def _embedd_tautomers(self, mol):
        pass