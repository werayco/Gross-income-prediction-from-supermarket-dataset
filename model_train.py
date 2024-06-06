from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

import pandas as pd
import json
from dataprocessing import DataProcessing
from typing import Dict

obj = DataProcessing()

trn_arr,tst_arr = obj.process_01("./train.csv","./test.csv")

x_train,y_train,x_test,y_test = trn_arr[:,:-1],trn_arr[:,-1],tst_arr[:,:-1],tst_arr[:,-1]


def model_trainer(x_train,y_train,x_test,y_test) -> Dict:

    scores = {}
    models = { "Linear_Regression": LinearRegression(),
                "K_Neighbors_Regressor": KNeighborsRegressor(n_neighbors=3),
                "Random_Forest": RandomForestRegressor(max_depth=20,n_estimators=20,criterion="squared_error"),
                "SVR": SVR(),
                "Decision_Tree": DecisionTreeRegressor(criterion="poisson")}
    
    for model_name,model in models.items():
            model_fitter = model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            r2_score_00 = r2_score(y_test,y_pred)
            scores[model_name] = r2_score_00
    return scores

scores = model_trainer(x_train,y_train,x_test,y_test)

with open("metrics.json","w") as output:
      json.dump(scores,output)
