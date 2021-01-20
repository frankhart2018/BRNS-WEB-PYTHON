from flask import Flask, request, render_template, redirect, flash, jsonify, send_from_directory
import glob
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.spatial.distance import jaccard, cosine

from colorize import colorize
from contrast import contrast
from gamma import gamma_correction
from brns_processing import BRNSProcessing
from predict import predict_function

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'my-secret-key'
app.config['SESSION_TYPE'] = 'filesystem'

filename_global = ""
brns_processing = ""

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3,stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=8192, out_features=4000)
        self.droput = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=4000, out_features=2000)
        self.droput = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=2000, out_features=500)
        self.droput = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=500, out_features=50)
        self.droput = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=50, out_features=2)

    def forward(self,x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        out = out.view(-1,8192)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc5(out)
        return out

model = CNN()
model.load_state_dict(torch.load('static/model/vanilla-cnn-colored.pth', map_location=torch.device('cpu')))

jaccard_vectors = {
    "type_1": np.load("jaccard-vectors/type-1.npy"),
    "type_2": np.load("jaccard-vectors/type-2.npy"),
    "type_3": np.load("jaccard-vectors/type-3.npy"),
    "type_4": np.load("jaccard-vectors/type-4.npy"),
    "type_5": np.load("jaccard-vectors/type-5.npy"),
    "type_6": np.load("jaccard-vectors/type-6.npy"),
    "type_7": np.load("jaccard-vectors/type-7.npy"),
    "inorganic": np.load("jaccard-vectors/inorganic.npy"),
    "organic": np.load("jaccard-vectors/organic.npy"),
    "metal": np.load("jaccard-vectors/metal.npy"),
}

@app.route('/', methods=['GET', 'POST'])
def index():

    if "inputSubmit" in request.form:

        if "inputFile" in request.files:
            file = request.files['inputFile']
            file_name = secure_filename(file.filename)
            filename = "static/original/" + file_name
            file.save(filename)
            np_data = np.loadtxt(filename, dtype=int)
            filename = "static/original/" + file_name.split(".")[0] + ".npy"
            np.save(filename, np_data)
            global filename_global
            filename_global = filename
            global brns_processing
            brns_processing = BRNSProcessing(filename_global)
            img = brns_processing.pc_img

            im_filename = "static/colorized/" + file.filename.split(".")[0] + ".png"
            matplotlib.image.imsave(im_filename, img)
            return render_template('index.html', upload=False, img=im_filename)
        else:
            flash("No image selected")
            return render_template('index.html', upload=True)

    return render_template('index.html', upload=True)

@app.route('/zeff', methods=['GET', 'POST'])
def zeff():

    if request.method == "POST":

        filename_orig = request.form['filename']
        filename = "static/original/" + os.path.basename(filename_orig).split(".")[0] + ".npy"

        s = int(round(float(request.form['x1'])))
        t = int(round(float(request.form['y1'])))
        u = request.form['x2']
        v = request.form['y2']

        res, le, he = colorize(filename, all=True)

        a = le.flatten()
        b = he.flatten()
        c = np.log(a)
        d = np.log(b)
        R = c/d

        zeff = 5.7*(R**2)-7.4*R+8
        zeff1 = np.reshape(zeff, le.shape)
        I = res

        w = int(s) + int(u)
        y = int(t) + int(v)

        g = zeff1[int(t):int(t)+4, int(s):int(s)+4]

        return jsonify({'zeff': np.mean(g)})

@app.route('/constrast', methods=['GET', 'POST'])
def constrast():

    if request.method == "POST":

        filename = request.form['filename']

        savePath = contrast(filename)

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

@app.route('/rgb', methods=['GET', 'POST'])
def rgb():

    if request.method == "POST":
        global filename_global
        global brns_processing

        brns_processing = BRNSProcessing(filename_global)
        img = brns_processing.pc_img

        im_filename = "static/colorized/" + os.path.basename(filename_global).split(".")[0] + ".png"
        matplotlib.image.imsave(im_filename, img)
        return jsonify({"img": im_filename})

@app.route('/gray', methods=['GET', 'POST'])
def gray():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        img = colorize(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filename = os.path.basename(filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        im_filename = "static/grayscale/" + filename.split(".")[0] + ".png"
        cv2.imwrite(im_filename, img_gray)
        return jsonify({"img": im_filename})

@app.route('/hsi', methods=['GET', 'POST'])
def hsi():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        hsi_img = brns_processing.genHSIImg()
        im_filename = "static/hsi/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis('off')
        plt.imshow(hsi_img)
        plt.savefig(im_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})

@app.route('/cc', methods=['GET', 'POST'])
def cc():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        cc_img = brns_processing.genCCImg()
        filename_img = "static/colorized/" + os.path.basename(filename).split(".")[0] + ".png"
        im_real = cv2.imread(filename_img)
        shape_tuple = (im_real.shape[0], im_real.shape[1])
        cc_img = cv2.resize(cc_img, shape_tuple)
        im_filename = "static/cc/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis('off')
        plt.imshow(cc_img)
        plt.savefig(im_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})

@app.route('/inv', methods=['GET', 'POST'])
def inv():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        inv_img = brns_processing.genInvImg()
        im_filename = "static/inv/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis('off')
        plt.imshow(inv_img)
        plt.savefig(im_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})

@app.route('/obj', methods=['GET', 'POST'])
def obj():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        obj_img = brns_processing.genOvsBImg()
        im_filename = "static/obj/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis('off')
        plt.imshow(obj_img)
        plt.savefig(im_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})

@app.route('/om', methods=['GET', 'POST'])
def om():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        om_img = brns_processing.genOMImg()
        im_filename = "static/om/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis('off')
        plt.imshow(om_img)
        plt.savefig(im_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})

@app.route('/im', methods=['GET', 'POST'])
def im():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        im_img = brns_processing.genIMImg()
        im_filename = "static/im/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis('off')
        plt.imshow(im_img)
        plt.savefig(im_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})

@app.route('/vcminus', methods=['GET', 'POST'])
def vcminus():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        vcminus_img = brns_processing.genVCminus()
        im_filename = "static/vcminus/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis('off')
        plt.imshow(vcminus_img)
        plt.savefig(im_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})

@app.route('/vcplus', methods=['GET', 'POST'])
def vcplus():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        vcplus_img = brns_processing.genVCplus()
        im_filename = "static/vcplus/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis('off')
        plt.imshow(vcplus_img)
        plt.savefig(im_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})

@app.route('/gamma', methods=['GET', 'POST'])
def gamma():

    if request.method == "POST":
        global filename_global
        filename = "static/colorized/" + os.path.basename(filename_global).split(".")[0] + ".png"
        gamma = request.form['gamma']
        savePath = gamma_correction(gamma, filename)
        return jsonify({"img": savePath, "gamma": gamma})

@app.route('/ve', methods=['GET', 'POST'])
def ve():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing

        ve_val = round(float(request.form['ve']), 2)

        ve_img = brns_processing.genVEImg(ve_val)
        im_filename = "static/ve/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis('off')
        plt.imshow(ve_img)
        plt.savefig(im_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename, "ve": round(float(ve_val), 2)})

@app.route('/vd', methods=['GET', 'POST'])
def vd():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        vd_val = float(request.form['vd'])

        vd_img = brns_processing.genVDImg(vd_val)
        im_filename = "static/vd/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis('off')
        plt.imshow(vd_img)
        plt.savefig(im_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename, "vd": round(float(vd_val), 2)})

@app.route('/predict', methods=['POST'])
def predict():

    if request.method == "POST":
        filename = request.form['filename']
        im = cv2.imread(filename)

        s = int(round(float(request.form['x1'])))
        t = int(round(float(request.form['y1'])))
        u = request.form['x2']
        v = request.form['y2']

        w = int(s) + int(u)
        y = int(t) + int(v)

        g = im[int(t):int(t)+64, int(s):int(s)+64]
        file_save_name = "static/nn/" + filename.split("/")[-1].split(".")[0] + "_" + str(time.time()) + ".jpg"
        cv2.imwrite(file_save_name, g)

        prediction = predict_function(file_save_name, model)
        icon = "success"

        if(prediction == "Explosive"):
            icon = "error"

        return jsonify({'icon': icon, 'prediction': prediction})

@app.route('/jaccard', methods=['POST'])
def jaccard_index():

    if request.method == "POST":
        filename = request.form['filename']
        im = cv2.imread(filename)

        s = int(round(float(request.form['x1'])))
        t = int(round(float(request.form['y1'])))
        u = request.form['x2']
        v = request.form['y2']

        g = im[int(t):int(t)+10, int(s):int(s)+10]
        file_save_name = "static/jaccard/" + filename.split("/")[-1].split(".")[0] + "_" + str(time.time()) + ".png"
        cv2.imwrite(file_save_name, g)

        g = g.ravel()
        
        max_score = -1000
        max_type = ""
        for type_, vector in jaccard_vectors.items():
            assert g.shape == vector.shape
            score = jaccard(g, vector)

            if score > max_score:
                max_score = score
                max_type = type_

        return jsonify({'icon': 'success', 'status': f"Type: {max_type}, with score: {max_score}"})

@app.route('/cosine', methods=['POST'])
def cosine_similarity():

    if request.method == "POST":
        filename = request.form['filename']
        im = cv2.imread(filename)

        s = int(round(float(request.form['x1'])))
        t = int(round(float(request.form['y1'])))
        u = request.form['x2']
        v = request.form['y2']

        g = im[int(t):int(t)+10, int(s):int(s)+10]
        file_save_name = "static/cosine/" + filename.split("/")[-1].split(".")[0] + "_" + str(time.time()) + ".png"
        cv2.imwrite(file_save_name, g)

        g = g.ravel()
        
        max_score = -1000
        max_type = ""
        for type_, vector in jaccard_vectors.items():
            assert g.shape == vector.shape
            score = cosine(g, vector)

            if score > max_score:
                max_score = score
                max_type = type_

        return jsonify({'icon': 'success', 'status': f"Type: {max_type}, with score: {max_score}"})

@app.route('/download')
def download():

    return send_from_directory('static', 'model/vanilla-cnn-colored.pth')

if __name__ == "__main__":
    app.run()
    plt.close()
