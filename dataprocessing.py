from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
from dataclasses import dataclass
import pandas as pd
from logger import logging
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np

@dataclass
class OutputPath:
    train_path = os.path.join(os.getcwd(), "train_array.csv")
    test_path = os.path.join(os.getcwd(), "test_array.csv")

class DataProcessing:
    def __init__(self):
        self.output = OutputPath()

    def process(self):
        logging.info("Setting up data cleaning pipelines.")

        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")), 
            ("standard", StandardScaler(with_mean=False))
        ])
        
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder()),
            ("scaler", StandardScaler(with_mean=False))
        ])

        num_cols = ["Unit price", "Tax 5%", "gross margin percentage"]
        cat_cols = ["Branch", "City", "Gender"]

        prepr = ColumnTransformer([
            ("numerical", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ])

        return prepr
    
    def process_01(self, train_path, test_path):
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        features = ["Branch", "City", "Gender", "Unit price", "Tax 5%", "gross margin percentage"]
        target = ["gross income"]

        train_feat = train_data[features]
        test_feat = test_data[features]

        train_target = train_data[target]
        test_target = test_data[target]

        logging.info(f"The features of the project are {features} and the target is {target}")

        prp_obj = self.process()

        trn_arr_feat = prp_obj.fit_transform(train_feat)
        tst_arr_feat = prp_obj.transform(test_feat)

        # Impute and scale target variable
        imp_target = SimpleImputer(strategy="mean")
        scaler = StandardScaler(with_mean=False)

        train_tar_imputed = imp_target.fit_transform(train_target)
        test_tar_imputed = imp_target.transform(test_target)

        final_train_tar = scaler.fit_transform(train_tar_imputed)
        final_test_tar = scaler.transform(test_tar_imputed)

        train = np.c_[trn_arr_feat, final_train_tar]
        test = np.c_[tst_arr_feat, final_test_tar]

        df_train = pd.DataFrame(train)
        df_train.to_csv(self.output.train_path, header=True, index=False)

        df_test = pd.DataFrame(test)
        df_test.to_csv(self.output.test_path, header=True, index=False)

        return (train, test)


obj = DataProcessing()
trn_arr, tst_arr = obj.process_01("./train.csv", "./test.csv")
print(tst_arr)
