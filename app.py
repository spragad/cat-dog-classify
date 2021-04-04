
from keras.preprocessing import image
from tensorflow import keras
import numpy as np
import os

from flask import Flask, redirect, render_template, request, session, url_for
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

app = Flask(__name__)
dropzone = Dropzone(app)

app.config['SECRET_KEY'] = 'secretkey'

app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'predict'
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd()

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

model=keras.models.load_model("Trained Model")

@app.route("/", methods=['GET', 'POST'])
def home():
    
    # set session for image results
    if "file_urls" not in session:
        session['file_urls'] = []
    # list to hold our uploaded image urls
    file_urls = session['file_urls']
    
    # handle image upload from Dropszone
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            # save the file with to our photos folder
            filename = photos.save(file,name=file.filename)
            # append image urls
            file_urls.append(photos.url(filename))
        session['file_urls'] = file_urls
        return "uploading..."
    # return dropzone template on GET request  
    return render_template("home.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    # set session for pred results
    if "pred" not in session:
        session['pred'] = []
    # list to hold our uploaded image urls
    pred = session['pred']
    
    # redirect to home if no images to display
    if "file_urls" not in session or session['file_urls'] == []:
        return redirect(url_for('home'))

    # set the file_urls and remove the session variable
    file_urls = session['file_urls']
    pred= session['pred']

    for pic in os.listdir(os.getcwd()):
        img_width, img_height = 200, 200
        img = image.load_img(pic, target_size = (img_width, img_height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        out=model.predict_classes(img)
        if out==0:
            pred.append("It's a DOG!")
        else:
            pred.append("It's a CAT!")
    
    session["pred"]=pred
    
    session.pop('file_urls', None)
    session.pop('pred', None)
    
    return render_template('predict.html', file_urls=file_urls, pred=pred)

@app.route("/clear", methods=['GET', 'POST'])
def clear():
    if "pred" in session:
        session['pred'] = []
        
    if len(os.listdir(os.getcwd()))>0:
        for pic in os.listdir(os.getcwd()):
            os.remove(os.getcwd()+'/'+pic)
    return redirect(url_for('home'))

if __name__ == '__main__':
   app.run(debug = True)