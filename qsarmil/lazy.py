# ==========================================================
# Utility Functions
# ==========================================================
def write_model_predictions(model_name: str,
                            smiles_list: List[str],
                            y_true: List[Any],
                            y_pred: List[Any],
                            output_path: str) -> None:
    """Append or write predictions of a model to a CSV file safely.

    If the file exists, adds or replaces a column for `model_name`.
    Preserves existing column order. Thread-safe via FileLock.

    Args:
        model_name (str): Name of the model / column to write.
        smiles_list (List[str]): SMILES strings corresponding to predictions.
        y_true (List[Any]): True labels.
        y_pred (List[Any]): Predicted labels.
        output_path (str): CSV file path to write predictions to.
    """
    ...


def replace_nan_with_column_mean(bags: List[np.ndarray]) -> List[np.ndarray]:
    """Replace NaNs in bags with column-wise mean across all instances.

    Args:
        bags (List[np.ndarray]): List of 2D arrays representing bags of instances.

    Returns:
        List[np.ndarray]: Bags with NaNs replaced by column means.
    """
    ...


# ==========================================================
# Descriptor & Conformer helpers
# ==========================================================
def gen_conformers(smi_list: List[str], n_cpu: int = 1) -> List[Any]:
    """Generate 3D conformers for a list of SMILES strings using RDKit.

    Args:
        smi_list (List[str]): List of SMILES strings.
        n_cpu (int, optional): Number of CPU threads for parallel conformer generation.

    Returns:
        List[Any]: List of RDKit molecules with generated conformers or FailedConformer objects.
    """
    ...


def calc_descriptors(descriptor: DescriptorWrapper, df_data: pd.DataFrame, conf: Optional[List[Any]] = None) -> Tuple[List[str], List[np.ndarray], pd.Series]:
    """Calculate molecular descriptors for a dataset using a DescriptorWrapper.

    Args:
        descriptor (DescriptorWrapper): Descriptor computation object.
        df_data (pd.DataFrame): Dataset with SMILES in first column and labels in second.
        conf (Optional[List[Any]]): Pre-generated conformers corresponding to SMILES.

    Returns:
        Tuple[List[str], List[np.ndarray], pd.Series]: SMILES list, descriptor arrays (bags), and labels.
    """
    ...


# ==========================================================
# ModelBuilder Class
# ==========================================================
class MILBuilder:
    """Encapsulates training and prediction of a single MIL model.

    Combines a descriptor and an estimator, optionally performs hyperparameter optimization,
    scales the descriptors, trains the estimator, and writes predictions to CSV.
    """

    def __init__(self,
                 descriptor_name: str,
                 descriptor_obj: DescriptorWrapper,
                 estimator: Any,
                 hopt: bool,
                 model_name: str,
                 model_folder: str,
                 n_cpu: int = 1):
        """Initialize a MILBuilder instance.

        Args:
            descriptor_name (str): Name of the descriptor.
            descriptor_obj (DescriptorWrapper): Descriptor object.
            estimator (Any): MIL estimator (regressor or classifier).
            hopt (bool): Whether to perform hyperparameter optimization.
            model_name (str): Name of the model for output files.
            model_folder (str): Folder to save prediction CSV files.
            n_cpu (int): Number of CPU threads for computation.
        """
        ...

    def scale_descriptors(self, x_train: List[np.ndarray], x_val: List[np.ndarray], x_test: List[np.ndarray]):
        """Fit BagMinMaxScaler on training bags and transform train/val/test sets.

        Args:
            x_train (List[np.ndarray]): Training descriptor bags.
            x_val (List[np.ndarray]): Validation descriptor bags.
            x_test (List[np.ndarray]): Test descriptor bags.

        Returns:
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]: Scaled train, val, and test bags.
        """
        ...

    def run(self, desc_dict: Dict[str, Dict[str, Tuple[List[str], List[np.ndarray], pd.Series]]]):
        """Execute the training and prediction pipeline for this model.

        Steps:
          1. Retrieve SMILES, descriptors, and labels from desc_dict.
          2. Scale descriptors.
          3. Optionally perform hyperparameter optimization.
          4. Train the estimator.
          5. Predict on validation and test sets.
          6. Write predictions to CSV.

        Args:
            desc_dict (Dict[str, Dict[str, Tuple[List[str], List[np.ndarray], pd.Series]]]):
                Precomputed descriptors and labels for train/val/test.

        Returns:
            MILBuilder: Returns self for convenience.
        """
        ...


# ==========================================================
# LazyMIL Class
# ==========================================================
class LazyMIL:
    """Lightweight orchestrator for MIL modeling.

    Automates conformer generation, descriptor calculation, and MIL model training
    for multiple descriptors and estimators.
    """

    def __init__(self, task: str = "regression", hopt: bool = False, output_folder: Optional[str] = None, n_cpu: int = 1, verbose: bool = True):
        """Initialize LazyMIL orchestrator.

        Args:
            task (str): Either 'regression' or 'classification'.
            hopt (bool): Whether to perform hyperparameter optimization.
            output_folder (Optional[str]): Folder to save model predictions.
            n_cpu (int): Number of CPU threads for computation.
            verbose (bool): Whether to print progress messages.
        """
        ...

    def run(self, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
        """Execute the full pipeline for train/val/test datasets.

        Steps:
          1. Generate conformers for each dataset split.
          2. Compute all descriptors for each dataset split.
          3. Instantiate MILBuilder for every (descriptor, estimator) pair.
          4. Train each model and write predictions to CSV.

        Args:
            df_train (pd.DataFrame): Training dataset with SMILES and labels.
            df_val (pd.DataFrame): Validation dataset with SMILES and labels.
            df_test (pd.DataFrame): Test dataset with SMILES and labels.

        Returns:
            LazyMIL: Returns self for convenience.
        """
        ...
