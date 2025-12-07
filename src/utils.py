import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill 
from sklearn.metrics import r2_score
def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise Exception(e,sys)


def evaluate_models(X_train,Y_train,X_test,Y_test,models):
    try:
        report={}
        for model_name,model in models.items():
            model.fit(X_train,Y_train)
            Y_predict=model.predict(X_test)
            score=r2_score(Y_test,Y_predict)
            report[model_name]=score
            
        return report
    except:
        pass