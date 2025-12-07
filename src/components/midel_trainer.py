import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd 
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
from catboost import CatBoostRegressor
@dataclass
class ModelTrainerConfig:
    model_trainer_file_path=os.path.join('artfacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Splitting training and test input data')
            X_train,Y_train,X_test,Y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
                )
            
            models={
                'Random Forest':RandomForestRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                "Linear Regressor":LinearRegression(),
                'K-Neighbores Regressions':KNeighborsRegressor(),
                'XGBoosting Regression':XGBRFRegressor(),
                'CatBoosting Regression':CatBoostRegressor(verbose=False),
                "AdaBoost Regression":AdaBoostRegressor(),
                'SVR':SVR()
            }
            
            # Hyper tuning for all models
            
            Parameter={
                
                "Random Forest":{ "max_features":[ "sqrt", "log2"],
                                 "criterion":["squared_error", "absolute_error"]
                    
                },
                
                "Decision Tree":{
                    "splitter":["best", "random"],
                    "criterion":["squared_error", "friedman_mse"]
                },
                
                'Gradient Boosting':{
                    "criterion":["friedman_mse", "squared_error"]
                },
                
                "Linear Regressor":{
                    
                },
                
                'K-Neighbores Regressions':{
                    "algorithm":["auto", "ball_tree", "kd_tree", "brute"],
                    "weights":["uniform", "distance"]
                },
                'XGBoosting Regression':{
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                },
                'CatBoosting Regression':{
                    'learning_rate': [0.03, 0.1],
                    'depth': [4, 6, 8],
                    'l2_leaf_reg': [1, 3]
                },
                "AdaBoost Regression":{
                            'n_estimators': [50, 100, 200],
                            'learning_rate': [0.01, 0.1, 1.0],
                },
                'SVR':{
                    'C': [0.1, 1, 10, 100],
                    'epsilon': [0.01, 0.1, 0.5],
                }
            }
            
        
            
            model_report:dict=evaluate_models(X_train=X_train,Y_train=Y_train,
                                              X_test=X_test,Y_test=Y_test,models=models,Parameter=Parameter)
            # Best model score
            best_model_score=max(sorted(model_report.values()))
            for mod_name,score in model_report.items():
                if  best_model_score==score:
                    best_model_name=mod_name # best model name
                
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info('Best found model on both training and dataset')
            
            
            
            save_object(
                file_path=self.model_trainer_config.model_trainer_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            r2_scor=r2_score(Y_test,predicted)
            
            return r2_scor
            
        except Exception as e:
            raise CustomException(e,sys)