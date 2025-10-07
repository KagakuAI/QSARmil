import os
import shutil
import tempfile
import hashlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from filelock import FileLock

from molfeat.calc import Pharmacophore3D, USRDescriptors, ElectroShapeDescriptors

from qsarmil.conformer import RDKitConformerGenerator
from qsarmil.descriptor.rdkit import (RDKitGEOM,
                                      RDKitAUTOCORR,
                                      RDKitRDF,
                                      RDKitMORSE,
                                      RDKitWHIM,
                                      RDKitGETAWAY)

from qsarmil.utils.logging import FailedConformer, FailedDescriptor
from qsarmil.descriptor.wrapper import DescriptorWrapper

# Preprocessing
from milearn.preprocessing import BagMinMaxScaler

# Network hparams
from milearn.network.module.hopt import DEFAULT_PARAM_GRID

# MIL network wrappers
from milearn.network.regressor import BagWrapperMLPNetworkRegressor, InstanceWrapperMLPNetworkRegressor
from milearn.network.classifier import BagWrapperMLPNetworkClassifier, InstanceWrapperMLPNetworkClassifier

# MIL networks
from milearn.network.regressor import (InstanceNetworkRegressor,
                                       BagNetworkRegressor,
                                       AdditiveAttentionNetworkRegressor,
                                       SelfAttentionNetworkRegressor,
                                       HopfieldAttentionNetworkRegressor,
                                       DynamicPoolingNetworkRegressor)

from milearn.network.classifier import (InstanceNetworkClassifier,
                                        BagNetworkClassifier,
                                        AdditiveAttentionNetworkClassifier,
                                        SelfAttentionNetworkClassifier,
                                        HopfieldAttentionNetworkClassifier,
                                        DynamicPoolingNetworkClassifier)


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
    "RDKitGETAWAY": DescriptorWrapper(RDKitGETAWAY()), # can be long
    "MolFeatPmapper": DescriptorWrapper(Pharmacophore3D(factory='pmapper')), # can be long
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
    "DynamicPoolingNetworkRegressor": DynamicPoolingNetworkRegressor()
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
    "DynamicPoolingNetworkClassifier": DynamicPoolingNetworkClassifier()
}


# ==========================================================
# Utility Functions
# ==========================================================
def write_model_predictions(model_name, smiles_list, y_true, y_pred, output_path):
    """Append new model predictions as a column to CSV assuming fixed row order."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lockfile = os.path.join(tempfile.gettempdir(), f"{hashlib.md5(output_path.encode()).hexdigest()}.lock")

    new_col = pd.DataFrame({model_name: y_pred})

    with FileLock(lockfile):
        if os.path.exists(output_path):
            df = pd.read_csv(output_path)
            df[model_name] = new_col[model_name]
        else:
            df = pd.DataFrame({
                "SMILES": smiles_list,
                "Y_TRUE": y_true,
                model_name: y_pred
            })

        # Optional: reorder columns for readability
        cols = ["SMILES", "Y_TRUE"] + sorted(c for c in df.columns if c not in {"SMILES", "Y_TRUE"})
        df = df[cols]
        df.to_csv(output_path, index=False)

def replace_nan_with_column_mean(bags):
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
# ModelBuilder Class
# ==========================================================
def gen_conformers(smi_list, n_cpu=1):
    """Load SMILES and properties from CSV."""

    mol_list = []
    for smi in smi_list:
        mol = Chem.MolFromSmiles(smi)
        mol_list.append(mol)

    conf_gen = RDKitConformerGenerator(num_conf=10, num_cpu=n_cpu, verbose=True)
    conf_list = conf_gen.run(mol_list)
    return conf_list

def calc_descriptors(descriptor, df_data, conf=None):
    """Load SMILES and properties from CSV."""

    smi, y = df_data.iloc[:, 0], df_data.iloc[:, 1]
    x = descriptor.transform(conf)
    x = replace_nan_with_column_mean(x)
    return smi, x, y

class MILBuilder:
    def __init__(self, descriptor, estimator, model_name, model_folder, n_cpu=1):
        self.descriptor = descriptor
        self.estimator = estimator
        self.model_name = model_name
        self.model_folder = model_folder
        self.n_cpu = n_cpu

    def scale_descriptors(self, x_train, x_val, x_test):
        scaler = BagMinMaxScaler()
        scaler.fit(x_train)
        return scaler.transform(x_train), scaler.transform(x_val), scaler.transform(x_test)

    def run(self, desc_dict):

        # 1. Get mol descriptors
        smi_train, x_train, y_train = desc_dict["df_train"][self.descriptor]
        smi_val, x_val, y_val = desc_dict["df_val"][self.descriptor]
        smi_test, x_test, y_test = desc_dict["df_test"][self.descriptor]

        # 2. Scale descriptors
        x_train_scaled, x_val_scaled, x_test_scaled = self.scale_descriptors(x_train, x_val, x_test)

        # 3. Train estimator
        estimator = self.estimator
        estimator.fit(x_train_scaled, y_train)

        # 4. Make val/test predictions
        pred_val = list(estimator.predict(x_val_scaled))
        pred_test = list(estimator.predict(x_test_scaled))

        # 5. Save predictions
        write_model_predictions(self.model_name, smi_val, y_val, pred_val,
                                os.path.join(self.model_folder, "val.csv"))

        write_model_predictions(self.model_name, smi_test, y_test, pred_test,
                                os.path.join(self.model_folder, "test.csv"))

        return self

class LazyMIL:
    def __init__(self, task="regression", output_folder=None, n_cpu=1, verbose=True):
        self.task = task
        self.output_folder = output_folder
        self.n_cpu = n_cpu
        self.verbose = verbose

        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
            os.makedirs(self.output_folder, exist_ok=True)

    def run(self, df_train, df_val, df_test):

        conf_dict = {"df_train": gen_conformers(smi_list=df_train.iloc[:, 0], n_cpu=self.n_cpu),
                     "df_val": gen_conformers(smi_list=df_val.iloc[:, 0], n_cpu=self.n_cpu),
                     "df_test": gen_conformers(smi_list=df_test.iloc[:, 0], n_cpu=self.n_cpu)}

        desc_dict = {"df_train": {}, "df_val":{}, "df_test":{}}

        all_models = []
        for desc_name, descriptor in DESCRIPTORS.items():

            desc_dict["df_train"][desc_name] = calc_descriptors(descriptor, df_train, conf=conf_dict["df_train"])
            desc_dict["df_val"][desc_name] = calc_descriptors(descriptor, df_val, conf=conf_dict["df_val"])
            desc_dict["df_test"][desc_name] = calc_descriptors(descriptor, df_test, conf=conf_dict["df_test"])

            for est_name, estimator in REGRESSORS.items():

                # 1. Create result folder
                model_name = f"{desc_name}|{est_name}"

                # 2. Create model
                model = MILBuilder(
                    descriptor=desc_name,
                    estimator=estimator,
                    model_name=model_name,
                    model_folder=self.output_folder,
                    n_cpu=self.n_cpu,
                )

                # 3. Add model
                all_models.append(model)

        n = 0
        for model in all_models:
            model.run(desc_dict)
            n += 1
            if self.verbose:
                print(f"{n} / {len(all_models)} / {model.model_name}", end="\r")

        return self