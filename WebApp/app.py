from flask import Flask,render_template,redirect,request,url_for 
from CaptionIt import predict_caption
import os

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html',title='CaptionIt')

@app.route('/upload',methods=['POST'])
def upload():
    path=None
    caption=None
    if request.method=='POST':
        if request.files.get('imgFile'):
            image = request.files['imgFile']
            image.save('static/images/'+image.filename)
            caption = predict_caption('static/images/'+image.filename)
            path=os.path.join('images/',image.filename)
        else:
            pass    
    return render_template('index.html',title='CaptionIt',imagePath=path,imgCaption=caption)

if __name__=='__main__':
    app.run()

