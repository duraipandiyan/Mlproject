import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
from src.utils import save_object


from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artfacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation=DataTransformationConfig()
        
    def get_data_transformation_object(self):
        
        "This funtion is responsible for data transformation"
        
        try:
            Numarical_columns=['reading_score', 
                               'writing_score']
            
            Categorical_columns=['gender', 
                                 'race_ethnicity', 
                                 'parental_level_of_education', 
                                 'lunch', 
                                 'test_preparation_course']
            
            Numarical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ('Scalar',StandardScaler())
                ]
            )
            
            Cate_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoding',OneHotEncoder()),
                    ('Scalar',StandardScaler(with_mean = False))
                ]
                
            )
            
            logging.info('categorical columns we have done one hot encoding, inputer and Scalar')
            logging.info('Numarical we have done scalar and imputer')
            
            preprocessor=ColumnTransformer(
                [
                    ('Numarical_pipeline',Numarical_pipeline,Numarical_columns),
                    ('Cate_pipeline',Cate_pipeline,Categorical_columns)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info('train and test data has been loaded')
            
            
            logging.info('Obtaining preprocessing object')
            
            preprocessing_obj=self.get_data_transformation_object()
            
            target_column_name="math_score"
            Numarical_columns=['reading_score', 
                               'writing_score']
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1) #X_train_data
            target_feature_train_df=train_df[target_column_name]   #Y_train_data
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1) # X_test_data
            target_feature_test_df=test_df[target_column_name] #Y_test_data
            
            logging.info(
                "Transform the input_feature_train_df and target_feature_train_df "
                
            )
            
            input_feature_train_arry=preprocessing_obj.fit_transform(input_feature_train_df) #X_train
            input_feature_test_arry=preprocessing_obj.transform(input_feature_test_df) #X_test
            
            
            # Your pipeline expects one final combined dataset as a NumPy array.
            # So train_arr = [X_train | y_train]
            # And test_arr = [X_test | y_test]
            
            train_arr = np.c_[
                input_feature_train_arry, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[input_feature_test_arry, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            
            save_object(
                
                file_path=self.data_transformation.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e,sys)
            
        
