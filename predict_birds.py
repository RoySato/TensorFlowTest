import os
from flask import Flask, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from keras.models import Sequential, load_model
import numpy as np
from PIL import Image

classes = ["redcrestedcardinal", "sparrow", "chicken"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = 'D:/Python/TensorFlow/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method=='POST':
        if 'file' not in request.files:
            flash('File was not found.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('File was not found.')
            return file and allowed_file(file.filename)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            model = load_model('./birds_cnn.h5')
            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            predicted = result.argmax()
            percentage = int(result[predicted] * 100)
            name = ""
            if classes[predicted] == "redcrestedcardinal":
                name = "コウカンチョウ"
            if classes[predicted] == "sparrow":
                name = "スズメ"
            if classes[predicted] == "chicken":
                name = "ニワトリ"

            return '{}%の確率で、{}です'.format(str(percentage), name)

    return '''
    <!doctype html>
    <html><head><title>ハワイの鳥判定器</title></head>
    <body>
    <meta charset="UTF-8">
    <h1>ハワイの鳥判定器</h1>
    <form method = post enctype = multipart/form-data>
    <p><input type=file name=file>
    <input type=submit value=Upload>
    </form>
    </body>
    </html>
    '''

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


