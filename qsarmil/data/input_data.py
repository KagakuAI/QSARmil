from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed
import pandas as pd

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


class DataValidator:
    """Fast parallel SMILES validator + 3D conformer check using RDKit."""

    def __init__(self, num_cpu: int = -1, verbose: bool = True):
        self.num_cpu = num_cpu
        self.verbose = verbose

    def _validate_one(self, smiles: str) -> dict:
        result = {
            "smiles": smiles,
            "is_valid_smiles": False,
            "sanitization_passed": False,
            "conformer_generated": False,
            "error": None
        }

        # Step 1 - SMILES parsing
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result["error"] = "SMILES parsing failed"
            return result

        result["is_valid_smiles"] = True

        # Step 2 - Molecule sanitization
        try:
            Chem.SanitizeMol(mol)
            result["sanitization_passed"] = True
        except Exception as e:
            result["error"] = f"Sanitization failed: {str(e)}"
            return result

        # Step 3 - Add hydrogens
        try:
            mol = Chem.AddHs(mol)
        except Exception as e:
            result["error"] = f"AddHs failed: {str(e)}"
            return result

        # Step 4 - Generate single conformer
        try:

            params = AllChem.ETKDGv3()
            params.randomSeed = 42

            conf_id = AllChem.EmbedMolecule(mol, params)
            if conf_id == -1:
                result["error"] = "Conformer embedding failed"
                return result
            result["conformer_generated"] = True
            return result

        except Exception as e:
            result["error"] = f"Embedding exception: {str(e)}"
            return result

    def validate_smiles(self, smiles_list):
        """Run parallel validation over a list of SMILES."""

        smiles_list = list(smiles_list)

        return Parallel(n_jobs=self.num_cpu)(
            delayed(self._validate_one)(smi)
            for smi in smiles_list
        )

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and filter a dataframe."""

        smiles = df.iloc[:, 0].tolist()
        results = self.validate_smiles(smiles)

        mask = [r["conformer_generated"]for r in results]

        removed_rows = []
        for i, (r, keep) in enumerate(zip(results, mask)):
            if not keep:
                removed_rows.append({
                    "row_index": i,
                    "smiles": r["smiles"],
                    "reason": r["error"]
                })

        # print report
        if removed_rows:
            if self.verbose:
                print("\nRemoved rows:")
                for r in removed_rows:
                    print(f"  > Row {r['row_index']}: {r['smiles']} -> {r['reason']}")
        else:
            if self.verbose:
                print("\nNo rows removed. All molecules valid.")

        # return filtered dataframe
        return df[mask].reset_index(drop=True)