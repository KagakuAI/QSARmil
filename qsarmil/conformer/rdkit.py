from rdkit import Chem
from rdkit.Chem import AllChem
from qsarmil.conformer.base import ConformerGenerator


class RDKitConformerGenerator(ConformerGenerator):
    """Generate RDKit 3D conformers for molecules using the ETKDG method.

    Inherits from ConformerGenerator and implements RDKit-specific molecule
    preparation and conformer embedding.

    Args:
        num_conf (int): Number of conformers to generate per molecule.
        e_thresh (float, optional): Energy threshold for filtering high-energy conformers.
        num_cpu (int): Number of CPU threads to use for parallel processing.
        verbose (bool): Whether to display a progress bar during generation.
    """

    def __init__(self, num_conf=10, e_thresh=None, num_cpu=1, verbose=True):
        """Initialize RDKitConformerGenerator with generation parameters.

        Args:
            num_conf (int): Number of conformers to generate per molecule.
            e_thresh (float, optional): Energy threshold for filtering high-energy conformers.
            num_cpu (int): Number of CPU threads to use for parallel processing.
            verbose (bool): Whether to display a progress bar during generation.
        """
        super().__init__(num_conf=num_conf, e_thresh=e_thresh, num_cpu=num_cpu, verbose=verbose)

    def _prepare_molecule(self, mol):
        """Prepare a molecule by adding explicit hydrogens.

        Args:
            mol (rdkit.Chem.Mol): Input molecule.

        Returns:
            rdkit.Chem.Mol: Molecule with explicit hydrogens added.
        """
        mol = Chem.AddHs(mol)
        return mol

    def _embedd_conformers(self, mol):
        """Embed multiple 3D conformers for a molecule using RDKit.

        Overrides the base method to use RDKit-specific parameters.

        Args:
            mol (rdkit.Chem.Mol): Molecule to embed conformers for.

        Returns:
            rdkit.Chem.Mol: Molecule with embedded conformers.
        """
        mol = self._prepare_molecule(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=self.num_conf, maxAttempts=700, randomSeed=42)
        return mol
