import os
from model import sample
from compiler import web_compiler
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)


print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path):
    return sample.predict_image(input_path=img_path)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('base.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        preds = model_predict(file_path)
        try:
            web_compiler.web_compile(preds[1])
        except:
            return preds[0]+'    !!! ERROR UNABLE TO COMPILE !!!   '
        return preds[0]
    return None


if __name__ == '__main__':
    app.run(debug=True)