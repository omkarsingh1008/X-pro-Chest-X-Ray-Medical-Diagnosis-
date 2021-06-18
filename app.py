import re
from flask import Flask, app,redirect,request,url_for,render_template
from flask import jsonify
import os
from tensorflow.keras.models import load_model
import cv2
import pandas as pd
UPLOAD_FOLDER = './upload_image'

app=Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def root():
    return render_template('index.html')
@app.route('/upload_image',methods=['POST','GET'])
def upload_img():
    if request.method=='POST':
        file = request.files['fileToUpload']
        if file:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    model=load_model('model1',compile=False)
    disease_names= ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']
    img = cv2.imread(UPLOAD_FOLDER+'/'+file.filename, cv2.IMREAD_COLOR)
    img = img/255
    img = cv2.resize(img, (320, 320))
    img = img.reshape((1,320,320,3))
    arr=model.predict(img)
    arr=[float(str(i)) for i in arr[0]]
    result=dict(zip(disease_names,arr))
    data = {'Disease_name': result.keys(),
        'probability': result.values()} 
    new = pd.DataFrame.from_dict(data)
    
  
    
    return render_template('upload.html',result=new.to_html())




if __name__=="__main__":
    app.run('localhost',8040,debug=True)