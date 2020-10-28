from flask import Flask, request, flash, render_template
from werkzeug.utils import secure_filename
import os
from modelFun import predict

app = Flask(__name__,template_folder='templates',static_folder='static')
app.config["DEBUG"] = False
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = 'static/images'

@app.route("/",methods=['GET', 'POST'])
#@app.route("/index",methods=['GET', 'POST'])
def show_index():
        return render_template("index1.html")


@app.route("/upload",methods=['GET','POST'])
def upload():
    return render_template("upload1.html",fname="")
    

@app.route("/upload_file",methods=['GET', 'POST'])
def upload_image():
        file = request.files['file']
        filename = secure_filename(file.filename)
        filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filePath)
        data = f"<img id='uploaded_img' class='upload_img form-control' src='/static/images/{filename}' alt='Uploaded Image'>\n\n\n{filename}"
        return data
        

@app.route("/result",methods=['GET', 'POST'])
def result():
    result=None
    solution=''
    filename = request.form.get('filename')
    result,solution = predict(filename)
    return render_template("result1.html",fname= filename, result= result, solution=solution)


if __name__ == '__main__':
    #app = adder_page()
    app.run()
