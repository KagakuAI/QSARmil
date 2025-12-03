import copy
import hashlib
import os
import time
import psutil
import shutil
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

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

from molfeat.trans import MoleculeTransformer
from qsarmil.conformer.rdkit import RDKitConformerGenerator
from qsarmil.descriptor.rdkit import RDKitAUTOCORR, RDKitGEOM, RDKitGETAWAY, RDKitMORSE, RDKitRDF, RDKitWHIM
from qsarmil.descriptor.wrapper import DescriptorWrapper
from qsarmil.descriptor.concat import DescriptorConcat
from .utils.logging import OutputSuppressor

# ==========================================================
# Configuration
# ==========================================================
DESCRIPTORS = {
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
def _worker(func, args, kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return {"error": repr(e)}

def run_in_subprocess(func, *args, **kwargs):
    with ProcessPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_worker, func, args, kwargs)
        result = future.result()

    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(result["error"])

    return result

def clean_descriptors(bags: List[np.ndarray]) -> List[np.ndarray]:
    """Replace NaN values in each bag's instances with the column means computed across all instances."""

    # Concatenate all instances from all bags into one 2D array
    all_instances = np.vstack(bags)

    # Compute column means ignoring NaNs
    col_means = np.nanmean(all_instances, axis=0)

    # Replace NaNs in each bag with the corresponding column mean
    cleaned_bags = []
    for bag in bags:
        bag = np.array(bag, dtype=float)  # Ensure float for NaN support
        idx = np.where(np.isnan(bag))
        bag[idx] = np.take(col_means, idx[1])
        cleaned_bags.append(bag)

    return cleaned_bags

def gen_conformers(smi_list, n_cpu=1):
    """Generate conformers for a list of SMILES strings using RDKitConformerGenerator."""
    mol_list = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        mol_list.append(mol)
    conf_gen = RDKitConformerGenerator(num_conf=10, num_cpu=n_cpu, verbose=False)
    conf_list = conf_gen.run(mol_list)
    return conf_list

def calc_descriptors(conf_list, calculator) :
    calculator.verbose = False
    x = calculator.run(conf_list)
    x = clean_descriptors(x)
    return x

def scale_descriptors(x_train, x_test):
    scaler = BagMinMaxScaler()
    scaler.fit(x_train)
    return scaler.transform(x_train), scaler.transform(x_test)

# ==========================================================
# ModelBuilder Class
# ==========================================================
def build_model(x_train, x_val, x_test, y_train, y_val, y_test, estimator_instance, hopt=True):

    # 1. Scale train/val descriptors
    x_train_scaled, x_val_scaled = scale_descriptors(x_train, x_val)

    # 2. Optimize hyperparameters
    if hopt:
        estimator_instance.hopt(x_train_scaled, y_train, param_grid=DEFAULT_PARAM_GRID, verbose=False)

    # 4. Train on train split only (not final training yet)
    estimator_instance.fit(x_train_scaled, y_train)
    pred_train = list(estimator_instance.predict(x_train_scaled))
    pred_val = list(estimator_instance.predict(x_val_scaled))

    # 5. Retrain model on full (train + val)
    x_full, y_full = np.vstack([x_train, x_val]), np.hstack([y_train, y_val])
    x_full_scaled, x_test_scaled = scale_descriptors(x_full, x_test)
    estimator_instance.fit(x_full_scaled, y_full)
    pred_test = list(estimator_instance.predict(x_test_scaled))

    return pred_train, pred_val, pred_test

class LazyMIL:

    def __init__(self, task="regression", hopt=True, output_folder=None, verbose=True):
        self.task = task
        self.hopt = hopt
        self.output_folder = output_folder
        self.verbose = verbose
        self.estimators_dict = REGRESSORS if self.task == "regression" else CLASSIFIERS

        if self.output_folder and os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)

    def run(self, df_train, df_val, df_test):

        # 1. Get data (smiles and prop)
        result_df_train = pd.DataFrame()
        smi_train, y_train = list(df_train.iloc[:, 0]), list(df_train.iloc[:, 1])
        result_df_train["SMILES"], result_df_train["Y_TRUE"] = smi_train, y_train

        result_df_val = pd.DataFrame()
        smi_val, y_val = list(df_val.iloc[:, 0]), list(df_val.iloc[:, 1])
        result_df_val["SMILES"], result_df_val["Y_TRUE"] = smi_val, y_val

        result_df_test = pd.DataFrame()
        smi_test, y_test = list(df_test.iloc[:, 0]), list(df_test.iloc[:, 1])
        result_df_test["SMILES"], result_df_test["Y_TRUE"] = smi_test, y_test

        # 2. Generate conformers
        if self.verbose:
            print("Generating conformers...")

        conf_train = gen_conformers(smi_train, n_cpu=20)
        conf_val = gen_conformers(smi_val, n_cpu=20)
        conf_test = gen_conformers(smi_test, n_cpu=20)

        total_models = len(DESCRIPTORS) * len(self.estimators_dict)
        current_model = 0

        # 3. Calculate descriptors
        for desc_name, desc_calc in DESCRIPTORS.items():

            if self.verbose:
                print(f"Calculating {desc_name} descriptors...")

            x_train = calc_descriptors(conf_train, desc_calc)
            x_val = calc_descriptors(conf_val, desc_calc)
            x_test = calc_descriptors(conf_test, desc_calc)

            # 4. Train models
            for est_name, estimator in self.estimators_dict.items():

                model_name = f"{desc_name}|{est_name}"
                current_model += 1
                if self.verbose:
                    print(f"[{current_model}/{total_models}] Running model: {model_name}", flush=True)

                start = time.time()
                with OutputSuppressor() as logger:
                    pred_train, pred_val, pred_test = run_in_subprocess(
                        build_model,
                        x_train,
                        x_val,
                        x_test,
                        y_train,
                        y_val,
                        y_test,
                        estimator,
                        self.hopt
                    )
                elapsed_min = (time.time() - start) / 60

                # 5. Write predictions
                result_df_train[model_name] = pred_train
                result_df_train.to_csv(os.path.join(self.output_folder, "train.csv"), index=False)

                result_df_val[model_name] = pred_val
                result_df_val.to_csv(os.path.join(self.output_folder, "val.csv"), index=False)

                result_df_test[model_name] = pred_test
                result_df_test.to_csv(os.path.join(self.output_folder, "test.csv"), index=False)

                if self.verbose:
                    process = psutil.Process()
                    mem_gb = process.memory_info().rss / (1024 ** 3)
                    print(f"  â†³ Finished in {elapsed_min:.2f} min | Memory usage: {mem_gb:.3f} GB")

        return None

