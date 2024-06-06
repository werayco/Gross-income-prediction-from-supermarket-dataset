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
    models = {
        "Linear_Regression": LinearRegression(
            fit_intercept=True, 
            normalize=False, 
            copy_X=True
        ),
        "K_Neighbors_Regressor": KNeighborsRegressor(
            n_neighbors=5, 
            weights='distance', 
            algorithm='auto', 
            leaf_size=30, 
            p=2, 
            metric='minkowski'
        ),
        "Random_Forest": RandomForestRegressor(
            n_estimators=100, 
            max_depth=None, 
            min_samples_split=2, 
            min_samples_leaf=1, 
            min_weight_fraction_leaf=0.0, 
            max_features='auto', 
            max_leaf_nodes=None, 
            min_impurity_decrease=0.0, 
            bootstrap=True, 
            oob_score=False, 
            n_jobs=-1, 
            random_state=None, 
            verbose=0, 
            warm_start=False, 
            ccp_alpha=0.0, 
            max_samples=None
        ),
        "SVR": SVR(
            kernel='rbf', 
            degree=3, 
            gamma='scale', 
            coef0=0.0, 
            tol=1e-3, 
            C=1.0, 
            epsilon=0.1, 
            shrinking=True, 
            cache_size=200, 
            verbose=False, 
            max_iter=-1
        ),
        "Decision_Tree": DecisionTreeRegressor(
            criterion='squared_error', 
            splitter='best', 
            max_depth=None, 
            min_samples_split=2, 
            min_samples_leaf=1, 
            min_weight_fraction_leaf=0.0, 
            max_features=None, 
            random_state=None, 
            max_leaf_nodes=None, 
            min_impurity_decrease=0.0, 
            ccp_alpha=0.0
        )
    }
    
    for model_name,model in models.items():
            model_fitter = model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            r2_score_00 = r2_score(y_test,y_pred)
            scores[model_name] = r2_score_00
    return scores

scores = model_trainer(x_train,y_train,x_test,y_test)

with open("metrics.json","w") as output:
      json.dump(scores,output)


max_acc = min(list(scores.values()))
best_model = list(scores.keys())[list(scores.values()).index(max_acc)]
with open("results.txt","w") as result:
      result.write(f"the maximum accuracy is {max_acc} and the best model is {best_model} ")
