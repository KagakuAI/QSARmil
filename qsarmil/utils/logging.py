from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


class FailedMolecule:
    """Represents a molecule for which SMILES parsing or initialization failed.

    Attributes:
        smiles (str): The SMILES string that failed to parse.
    """

    def __init__(self, smiles):
        """Initialize a FailedMolecule with the problematic SMILES.

        Args:
            smiles (str): SMILES string that failed parsing.
        """
        super().__init__()
        self.smiles = smiles

    def __str__(self):
        """Return a human-readable error message.

        Returns:
            str: Error message describing the parsing failure.
        """
        return f"{self.smiles} -> SMILES parsing failed"


class FailedConformer:
    """Represents a molecule for which conformer generation failed.

    Attributes:
        mol (rdkit.Chem.Mol): Molecule that failed conformer generation.
    """

    def __init__(self, mol):
        """Initialize a FailedConformer with the failed molecule.

        Args:
            mol (rdkit.Chem.Mol): Molecule that failed conformer generation.
        """
        super().__init__()
        self.mol = mol

    def __str__(self):
        """Return a human-readable error message.

        Returns:
            str: Error message describing the conformer generation failure.
        """
        smi = Chem.MolToSmiles(self.mol)
        return f"{smi} -> conformer generation failed"


class FailedDescriptor:
    """Represents a molecule for which descriptor calculation failed.

    Attributes:
        mol (rdkit.Chem.Mol): Molecule that failed descriptor calculation.
    """

    def __init__(self, mol):
        """Initialize a FailedDescriptor with the failed molecule.

        Args:
            mol (rdkit.Chem.Mol): Molecule that failed descriptor calculation.
        """
        super().__init__()
        self.mol = mol

    def __str__(self):
        """Return a human-readable error message.

        Returns:
            str: Error message describing the descriptor calculation failure.
        """
        smi = Chem.MolToSmiles(self.mol)
        return f"{smi} -> descriptor calculation failed"
