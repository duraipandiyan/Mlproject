import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipelie:
    def __init__(self):
        pass
    
    def predicts(self,features):
        try:
            model_path='artfacts\model.pkl'
            preprocessor_path='artfacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features) # processor obj already have preprocessed so we no need to fit_transform . we just need to transform so that we will get perfect mean value
            preds=model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData():
    def __init__(self,
                 gender,
                 race_ethnicity_of_eduction,
                 parental_level_of_eduction,
                 lunch,
                 test_preparation_course,
                 reading_score,
                 writing_score):
        
        self.gender= gender
        self.race_ethnicity_of_eduction=race_ethnicity_of_eduction
        self.parental_level_of_eduction=parental_level_of_eduction
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_data_as_data_frame(self):
        
        try:
            custom_data_input_dic={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity_of_eduction],
                'parental_level_of_education':[self.parental_level_of_eduction],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                "reading_score":[self.reading_score],
                'writing_score':[self.writing_score]
            }
            
            return pd.DataFrame(custom_data_input_dic)
        
        except Exception as e:
            raise CustomException(e,sys)
        
        