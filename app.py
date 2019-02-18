from flask import Flask, request, render_template, redirect, flash, jsonify
import glob
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os

import matplotlib.pyplot as plt

from colorize import colorize
from contrast import contrast
from gamma import gamma_correction

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'my-secret-key'
app.config['SESSION_TYPE'] = 'filesystem'

@app.route('/', methods=['GET', 'POST'])
def index():

    if "inputSubmit" in request.form:

        if "inputFile" in request.files:
            file = request.files['inputFile']
            filename = "static/original/" + secure_filename(file.filename)
            file.save(filename)
            img = colorize(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_filename = "static/colorized/" + file.filename.split(".")[0] + ".jpg"
            cv2.imwrite(im_filename, img)
            # plt.imshow(img)
            # plt.show()
            return render_template('index.html', upload=False, img=im_filename)
        else:
            flash("No image selected")
            return render_template('index.html', upload=True)

    return render_template('index.html', upload=True)

@app.route('/zeff', methods=['GET', 'POST'])
def zeff():

    if request.method == "POST":

        filename = request.form['filename']

        im = cv2.imread(filename)

        filename = "static/original/" + os.path.basename(filename).split(".")[0] + ".npy"

        res, le, he = colorize(filename, all=True)

        a = le
        b = he
        c = np.log(a)
        d = np.log(b)
        R = c/d
        zeff = -1.5*(R**2)+23*R-17
        zeff1 = np.reshape(zeff, le.shape)

        I = res

        s = request.form['x1']
        t = request.form['y1']
        u = request.form['x2']
        v = request.form['y2']

        w = int(s) + int(u)
        y = int(t) + int(v)

        print(t, y, s, w)

        g = zeff1[int(t):int(y), int(s):int(w)]
        return jsonify({'zeff': np.mean(g)})

@app.route('/constrast', methods=['GET', 'POST'])
def constrast():

    if request.method == "POST":

        filename = request.form['filename']

        savePath = contrast(filename)

        return jsonify({"img": savePath})

@app.route('/gamma', methods=['GET', 'POST'])
def gamma():

    if request.method == "POST":

        filename = request.form['filename']
        gamma = request.form['gamma']

        savePath = gamma_correction(gamma, filename)

        return jsonify({"img": savePath})

@app.route('/rgbtohsi', methods=['GET', 'POST'])
def rgbtohsi():

    if request.method == "POST":

        filename = request.form['filename']

        img = cv2.imread(filename)

        savePath = 'static/hsi/' + os.path.basename(filename)

        img_hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imwrite(savePath, img_hsi)

        return jsonify({"img": savePath})
