import pandas as pd
from sklearn.model_selection import train_test_split
from qsarmil.lazy import LazyMIL
from qsarcons.consensus import SystematicSearch, GeneticSearch

class MultiConformerModel:

    def __init__(self, num_conf=10, task="regression", hopt=False, output_folder=None, verbose=True):
        super().__init__()

        self.num_conf = num_conf
        self.task = task
        self.hopt = hopt
        self.output_folder = output_folder
        self.verbose = verbose

    def run_predict(self, df_train, df_test):

        # 1. Fill fake test prop
        if len(df_test.columns) == 1:
            df_test[1] = [None for i in df_test.index]

        # 2. Train/val split
        df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)

        # 3. Build multiple models
        lazy_ml = LazyMIL(task=self.task, hopt=self.hopt, output_folder=self.output_folder, verbose=self.verbose)
        lazy_ml.run(df_train, df_val, df_test)

        # 4. Load model predictions
        res_val = pd.read_csv(f"{self.output_folder}/val.csv")
        res_test = pd.read_csv(f"{self.output_folder}/test.csv")

        x_val, true_val = res_val.iloc[:, 2:], res_val.iloc[:, 1]
        x_test = res_test.iloc[:, 2:]

        # 5. Run genetic search
        print("Running consensus search")
        cons_search = GeneticSearch(cons_size="auto", metric="auto", n_iter=50)

        best_cons = cons_search.run(x_val, true_val)
        pred_test = cons_search.predict(x_test[best_cons])

        print(f"Best consensus: {best_cons}")

        return pred_test