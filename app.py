from flask import Flask, render_template
import sqlite3
from lr import *
from process import *
from knn import *
from svm import *
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/index')
def home():
    return index()


@app.route('/reduction')
def reduction():
    process()
    return render_template("index.html")


@app.route('/svm')
def svm():
    svm_main()
    return render_template("svm.html")


@app.route('/lr')
def lr():
    lr_main()
    return render_template("lr.html")


@app.route('/knn')
def knn():
    knn_main()
    return render_template("knn.html")


if __name__ == '__main__':
    app.run(debug=True)
