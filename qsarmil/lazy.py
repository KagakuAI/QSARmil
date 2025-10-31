import copy
import hashlib
import os
import shutil
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from filelock import FileLock
from milearn.network.classifier import (AdditiveAttentionNetworkClassifier, BagNetworkClassifier,
                                        BagWrapperMLPNetworkClassifier, DynamicPoolingNetworkClassifier,
                                        HopfieldAttentionNetworkClassifier, InstanceNetworkClassifier,
                                        InstanceWrapperMLPNetworkClassifier, SelfAttentionNetworkClassifier)
# Network hparams
# MIL networks
# MIL network wrappers
from milearn.network.regressor import (AdditiveAttentionNetworkRegressor, BagNetworkRegressor,
                                       BagWrapperMLPNetworkRegressor, DynamicPoolingNetworkRegressor,
                                       HopfieldAttentionNetworkRegressor, InstanceNetworkRegressor,
                                       InstanceWrapperMLPNetworkRegressor, SelfAttentionNetworkRegressor)
# Preprocessing
from milearn.preprocessing import BagMinMaxScaler
from milearn.network.module.hopt import DEFAULT_PARAM_GRID
from molfeat.calc import ElectroShapeDescriptors, Pharmacophore3D, USRDescriptors
from rdkit import Chem

from qsarmil.conformer.rdkit import RDKitConformerGenerator
from qsarmil.descriptor.rdkit import RDKitAUTOCORR, RDKitGEOM, RDKitGETAWAY, RDKitMORSE, RDKitRDF, RDKitWHIM
from qsarmil.descriptor.wrapper import DescriptorWrapper

# ==========================================================
# Configuration
# ==========================================================
DESCRIPTORS: Dict[str, DescriptorWrapper] = {
    "RDKitGEOM": DescriptorWrapper(RDKitGEOM()),
    "RDKitAUTOCORR": DescriptorWrapper(RDKitAUTOCORR()),
    "RDKitRDF": DescriptorWrapper(RDKitRDF()),
    "RDKitMORSE": DescriptorWrapper(RDKitMORSE()),
    "RDKitWHIM": DescriptorWrapper(RDKitWHIM()),
    "MolFeatUSRD": DescriptorWrapper(USRDescriptors()),
    "MolFeatElectroShape": DescriptorWrapper(ElectroShapeDescriptors()),
    "RDKitGETAWAY": DescriptorWrapper(RDKitGETAWAY()),  # can be long
    "MolFeatPmapper": DescriptorWrapper(Pharmacophore3D(factory="pmapper")),  # can be long
}

REGRESSORS = {
    "MeanBagWrapperMLPNetworkRegressor": BagWrapperMLPNetworkRegressor(pool="mean"),
    "MeanInstanceWrapperMLPNetworkRegressor": InstanceWrapperMLPNetworkRegressor(pool="mean"),
    # classic mil networks
    "MeanBagNetworkRegressor": BagNetworkRegressor(pool="mean"),
    "MeanInstanceNetworkRegressor": InstanceNetworkRegressor(pool="mean"),
    # attention mil networks
    "AdditiveAttentionNetworkRegressor": AdditiveAttentionNetworkRegressor(),
    "SelfAttentionNetworkRegressor": SelfAttentionNetworkRegressor(),
    "HopfieldAttentionNetworkRegressor": HopfieldAttentionNetworkRegressor(),
    # other mil networks
    "DynamicPoolingNetworkRegressor": DynamicPoolingNetworkRegressor(),
}

CLASSIFIERS = {
    "MeanBagWrapperMLPNetworkClassifier": BagWrapperMLPNetworkClassifier(pool="mean"),
    "MeanInstanceWrapperMLPNetworkClassifier": InstanceWrapperMLPNetworkClassifier(pool="mean"),
    # classic mil networks
    "MeanBagNetworkClassifier": BagNetworkClassifier(pool="mean"),
    "MeanInstanceNetworkClassifier": InstanceNetworkClassifier(pool="mean"),
    # attention mil networks
    "AdditiveAttentionNetworkClassifier": AdditiveAttentionNetworkClassifier(),
    "SelfAttentionNetworkClassifier": SelfAttentionNetworkClassifier(),
    "HopfieldAttentionNetworkClassifier": HopfieldAttentionNetworkClassifier(),
    # other mil networks
    "DynamicPoolingNetworkClassifier": DynamicPoolingNetworkClassifier(),
}


# ==========================================================
# Utility Functions
# ==========================================================
def write_model_predictions(
    model_name: str, smiles_list: List[str], y_true: List[Any], y_pred: List[Any], output_path: str
) -> None:
    """Append or add new model predictions as a column to a CSV file while
    preserving existing column order.

    - If `output_path` exists, the function will add `model_name` column (or replace it if present).
      It will **not** reorder existing columns (except to append the new one at the end if it wasn't present).
    - If `output_path` does not exist, it will be created with columns: SMILES, Y_TRUE, <model_name>.

    Thread-safe via FileLock on a temp-based lockfile derived from output_path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lockfile = os.path.join(tempfile.gettempdir(), f"{hashlib.md5(output_path.encode()).hexdigest()}.lock")

    new_col = pd.Series(y_pred, name=model_name)

    with FileLock(lockfile):
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            # if lengths mismatch, attempt safe behavior
            if len(df) != len(new_col):
                # If existing DF has SMILES, try to align by SMILES if possible, otherwise raise
                try:
                    tmp = pd.DataFrame({"SMILES": smiles_list, model_name: y_pred, "Y_TRUE_NEW": y_true})
                    df = df.merge(tmp[["SMILES", model_name]], on="SMILES", how="left")
                except Exception:
                    raise ValueError(
                        f"Length mismatch when writing {output_path}: existing {len(df)} vs new {len(new_col)}"
                    )
            else:
                df[model_name] = new_col.values
        else:
            df = pd.DataFrame({"SMILES": smiles_list, "Y_TRUE": y_true, model_name: y_pred})

        # Ensure SMILES and Y_TRUE remain first columns and preserve existing order for others.
        cols = [c for c in df.columns if c not in {"SMILES", "Y_TRUE"}]
        ordered = ["SMILES", "Y_TRUE"] + cols
        df = df[ordered]
        df.to_csv(output_path, index=False)


def replace_nan_with_column_mean(bags: List[np.ndarray]) -> List[np.ndarray]:
    """Replace NaN values in each bag's instances with the column means
    computed across all instances.

    NOTE: This function keeps the original behavior from your code. You asked to ignore the
    robustness fixes that handle the case where an entire column is NaN.
    """
    # Concatenate all instances from all bags into one 2D array
    all_instances = np.vstack(bags)

    # Compute column means ignoring NaNs
    col_means = np.nanmean(all_instances, axis=0)

    # Replace NaNs in each bag with the corresponding column mean
    cleaned_bags = []
    for bag in bags:
        bag = np.array(bag, dtype=float)  # Ensure float for NaN support
        inds = np.where(np.isnan(bag))
        bag[inds] = np.take(col_means, inds[1])
        cleaned_bags.append(bag)

    return cleaned_bags


# ==========================================================
# Descriptor & Conformer helpers
# ==========================================================
def gen_conformers(smi_list: List[str], n_cpu: int = 1) -> List[Any]:
    """Generate conformers for a list of SMILES strings using
    RDKitConformerGenerator."""
    mol_list = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        mol_list.append(mol)

    conf_gen = RDKitConformerGenerator(num_conf=10, num_cpu=n_cpu, verbose=True)
    conf_list = conf_gen.run(mol_list)
    return conf_list


def calc_descriptors(
    descriptor: DescriptorWrapper, df_data: pd.DataFrame, conf: Optional[List[Any]] = None
) -> Tuple[List[str], List[np.ndarray], pd.Series]:
    """Calculate descriptors using a DescriptorWrapper for a dataset table."""
    smi = list(df_data.iloc[:, 0])
    y = df_data.iloc[:, 1]
    x = descriptor.run(conf)
    x = replace_nan_with_column_mean(x)
    return smi, x, y


# ==========================================================
# ModelBuilder Class
# ==========================================================
class MILBuilder:
    """Encapsulates training and prediction of a single MIL model (descriptor +
    estimator)."""

    def __init__(
        self,
        descriptor_name: str,
        descriptor_obj: DescriptorWrapper,
        estimator: Any,
        hopt: bool,
        model_name: str,
        model_folder: str,
        n_cpu: int = 1,
    ):
        self.descriptor_name = descriptor_name
        self.descriptor_obj = descriptor_obj
        self.estimator = estimator
        self.hopt = hopt
        self.model_name = model_name
        self.model_folder = model_folder
        self.n_cpu = n_cpu

    def scale_descriptors(self, x_train: List[np.ndarray], x_val: List[np.ndarray], x_test: List[np.ndarray]):
        """Fit BagMinMaxScaler on x_train and transform train/val/test bags."""
        scaler = BagMinMaxScaler()
        scaler.fit(x_train)
        return scaler.transform(x_train), scaler.transform(x_val), scaler.transform(x_test)

    def run(self, desc_dict: Dict[str, Dict[str, Tuple[List[str], List[np.ndarray], pd.Series]]]):
        """
        Execute the training pipeline for this model:
         1. retrieve smi/x/y for train/val/test from desc_dict,
         2. scale descriptors,
         3. optionally run hyperparameter optimization,
         4. fit estimator,
         5. predict on val/test,
         6. write predictions to CSV.

        Returns self for convenience.
        """
        # 1. Get mol descriptors using descriptor_name as key
        smi_train, x_train, y_train = desc_dict["df_train"][self.descriptor_name]
        smi_val, x_val, y_val = desc_dict["df_val"][self.descriptor_name]
        smi_test, x_test, y_test = desc_dict["df_test"][self.descriptor_name]

        # 2. Scale descriptors
        x_train_scaled, x_val_scaled, x_test_scaled = self.scale_descriptors(x_train, x_val, x_test)

        # 3. Train estimator — allow estimators that implement hopt
        estimator = self.estimator
        if self.hopt and hasattr(estimator, "hopt"):
            estimator.hopt(x_train_scaled, y_train, param_grid=DEFAULT_PARAM_GRID, verbose=False)
        estimator.fit(x_train_scaled, y_train)

        # 4. Make val/test predictions
        pred_val = list(estimator.predict(x_val_scaled))
        pred_test = list(estimator.predict(x_test_scaled))

        # 5. Save predictions
        val_path = os.path.join(self.model_folder, "val.csv")
        test_path = os.path.join(self.model_folder, "test.csv")

        write_model_predictions(self.model_name, smi_val, y_val, pred_val, val_path)
        write_model_predictions(self.model_name, smi_test, y_test, pred_test, test_path)

        return self


class LazyMIL:
    """
    Lightweight orchestrator that:
      - generates conformers for train/val/test SMILES,
      - computes descriptors for every descriptor in DESCRIPTORS,
      - builds a MILBuilder for each (descriptor, estimator) pair,
      - trains each model and writes predictions.
    """

    def __init__(
        self,
        task: str = "regression",
        hopt: bool = False,
        output_folder: Optional[str] = None,
        n_cpu: int = 1,
        verbose: bool = True,
    ):
        self.task = task
        self.hopt = hopt
        self.output_folder = output_folder
        self.n_cpu = n_cpu
        self.verbose = verbose

        if not self.output_folder:
            raise ValueError("output_folder must be specified and non-empty")

        # Safely recreate the output folder
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder, exist_ok=True)

    def run(self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
        """Main entry point."""
        # 1. Generate conformers for each split
        conf_dict = {
            "df_train": gen_conformers(smi_list=list(df_train.iloc[:, 0]), n_cpu=self.n_cpu),
            "df_val": gen_conformers(smi_list=list(df_val.iloc[:, 0]), n_cpu=self.n_cpu),
            "df_test": gen_conformers(smi_list=list(df_test.iloc[:, 0]), n_cpu=self.n_cpu),
        }

        # 2. Compute descriptors for every descriptor in DESCRIPTORS for each split
        desc_dict: Dict[str, Dict[str, Tuple[List[str], List[np.ndarray], pd.Series]]] = {
            "df_train": {},
            "df_val": {},
            "df_test": {},
        }

        # Select estimator pool based on task
        estimator_pool = REGRESSORS if self.task == "regression" else CLASSIFIERS

        all_models: List[MILBuilder] = []

        for desc_name, descriptor in DESCRIPTORS.items():
            desc_dict["df_train"][desc_name] = calc_descriptors(descriptor, df_train, conf=conf_dict["df_train"])
            desc_dict["df_val"][desc_name] = calc_descriptors(descriptor, df_val, conf=conf_dict["df_val"])
            desc_dict["df_test"][desc_name] = calc_descriptors(descriptor, df_test, conf=conf_dict["df_test"])

            for est_name, estimator in estimator_pool.items():
                # instantiate/copy estimator to avoid state sharing between models
                estimator_copy = copy.deepcopy(estimator)

                # create model name and MILBuilder instance
                model_name = f"{desc_name}|{est_name}"
                model = MILBuilder(
                    descriptor_name=desc_name,
                    descriptor_obj=descriptor,
                    estimator=estimator_copy,
                    hopt=self.hopt,
                    model_name=model_name,
                    model_folder=self.output_folder,
                    n_cpu=self.n_cpu,
                )
                all_models.append(model)

        # 3. Run training for all models sequentially
        n = 0
        total = len(all_models)
        for model in all_models:
            try:
                model.run(desc_dict)
            except Exception as exc:
                # Log error but continue with remaining models
                warnings.warn(f"Model {model.model_name} failed with error: {exc}")
            n += 1
            if self.verbose:
                print(f"{n} / {total} — {model.model_name}", end="\r")

        if self.verbose:
            print()  # newline after progress

        return self
