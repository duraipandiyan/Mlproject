from flask import Flask, request,render_template
import numpy as np
import pandas as pd
from src.pipline.predic_pipeline import CustomData,PredictPipelie
from sklearn.preprocessing import StandardScaler

Application=Flask(__name__)
app=Application


# router for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')

    else:
        data=CustomData(
            
                gender=request.form.get('gender'),
                race_ethnicity_of_eduction=request.form.get('ethnicity'),
                parental_level_of_eduction=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
        )
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipline=PredictPipelie()
        result=predict_pipline.predicts(pred_df)
        
        return render_template('home.html',result=result[0])
    
    
if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)