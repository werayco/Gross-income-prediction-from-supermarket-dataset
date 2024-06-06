import pandas as pd
import dill
from sklearn.metrics import r2_score
import logging
import sys
import os
import pickle as plkk
from exceptor import CustomException
from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV,KFold


def trans_data_pickle(file_path,name_of_object_to_be_saved):
    try:
        with open(file_path,"wb") as file_path:
            dill.dump(name_of_object_to_be_saved,file_path)

    except Exception as e:
        raise CustomException(e,sys)
        


def best_model(x_train, y_train, x_test, y_test, models, params):
    try:

        model_plus_scores = {}

        for model_name, model in models.items():
            # Extract parameters specific to the current model
            model_params = params.get(model_name) 

            gs = GridSearchCV(model, model_params, cv=3)
            gs.fit(x_train, y_train)

            # Set best parameters found during grid search
            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            # Evaluate the model on test data
            y_pred = model.predict(x_test)
            
            r2scores = r2_score(y_true=y_test, y_pred=y_pred)

            model_plus_scores[model_name] = r2scores

        return model_plus_scores
    except Exception as e:
        raise CustomException(e,sys)