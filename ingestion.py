import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from dataclasses import dataclass
import os
import sys
from collections import deque


from exceptor import CustomException
@dataclass
class OutputPath:
    train_path = os.path.join(os.getcwd(),"./train.csv")
    test_path = os.path.join(os.getcwd(),"./test.csv")

class DataIngestion:
    def __init__(self):
        self.output = OutputPath()


    def process(self, data):
        try:
            all_data = pd.read_csv(data)
            folder = KFold(n_splits=2,random_state=40)

            for train_in, test_in in folder.split([all_data]):
                train_data, test_data = all_data.iloc[train_in],all_data[test_in]
            # train_data, test_data = train_test_split(all_data, random_state=45, train_size=0.7)
            train_data.to_csv(self.output.train_path, header=True, index=False)
            test_data.to_csv(self.output.test_path, header=True, index=False)

        except Exception as e:
            CustomException(e,sys)

if "__main__" == __name__:
    ing_obj = DataIngestion()
    ing_obj.process(".\supermarket_sales.csv")
