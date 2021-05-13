from flask import (
    Flask,
    request,
    render_template,
    redirect,
    flash,
    jsonify,
    send_from_directory,
)
import glob
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
import matplotlib
import time

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import jaccard, cosine

from colorize import colorize
from contrast import contrast
from gamma import gamma_correction
from brns_processing import BRNSProcessing
from predict import predict_function
from db import (
    get_connection,
    update_mode,
    update_scan_count,
    get_current_noobj_file_path,
    update_current_noobj_file_path,
)
from decorators import timeit

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

app = Flask(__name__)

app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = "my-secret-key"
app.config["SESSION_TYPE"] = "filesystem"

filename_global = ""
brns_processing = ""

jaccard_vectors = {
    "type_1": np.load("jaccard-vectors/type-1.npy"),
    "type_2": np.load("jaccard-vectors/type-2.npy"),
    "type_3": np.load("jaccard-vectors/type-3.npy"),
    "type_4": np.load("jaccard-vectors/type-4.npy"),
    "type_5": np.load("jaccard-vectors/type-5.npy"),
    "type_6": np.load("jaccard-vectors/type-6.npy"),
    "type_7": np.load("jaccard-vectors/type-7.npy"),
    # "inorganic": np.load("jaccard-vectors/inorganic.npy"),
    "organic": np.load("jaccard-vectors/organic.npy"),
    # "metal": np.load("jaccard-vectors/metal.npy"),
}

gamma = 0.1


@app.route("/", methods=["GET", "POST"])
@timeit
def index():

    if "inputSubmit" in request.form:

        if "inputFile" in request.files:
            file = request.files["inputFile"]
            file_name = secure_filename(file.filename)
            filename = "static/original/" + file_name
            file.save(filename)
            np_data = np.loadtxt(filename, dtype=int)
            filename = "static/original/" + file_name.split(".")[0] + ".npy"
            np.save(filename, np_data)
            global filename_global
            filename_global = filename
            global brns_processing

            noObjFile = get_current_noobj_file_path()
            if "inputNoObjFile" in request.files:
                no_object_file = request.files["inputNoObjFile"]

                if len(no_object_file.filename) > 0:
                    no_object_file_path = secure_filename(no_object_file.filename)
                    no_object_file_path = "static/noobj/" + no_object_file_path
                    no_object_file.save(no_object_file_path)

                    noObjFile = no_object_file_path
                    update_current_noobj_file_path(file_path=noObjFile)

            brns_processing = BRNSProcessing(filename_global, noobj_path=noObjFile)
            color_img = brns_processing.pc_img

            im_filename = "static/colorized/" + file.filename.split(".")[0] + ".png"
            matplotlib.image.imsave(im_filename, color_img)

            om_img = brns_processing.genOMImg()
            om_filename = (
                "static/om/" + os.path.basename(filename.split(".")[0]) + ".png"
            )
            plt.axis("off")
            plt.imshow(om_img)
            plt.savefig(om_filename, bbox_inches="tight", pad_inches=0)
            plt.close()

            img = cv2.imread(im_filename)
            filename = os.path.basename(im_filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            gray_filename = "static/grayscale/" + filename.split(".")[0] + ".png"
            cv2.imwrite(gray_filename, img_gray)

            vcplus_img = brns_processing.genVCplus()
            vc_filename = (
                "static/vcplus/" + os.path.basename(filename.split(".")[0]) + ".png"
            )
            plt.axis("off")
            plt.imshow(vcplus_img)
            plt.savefig(vc_filename, bbox_inches="tight", pad_inches=0)
            plt.close()

            conn = get_connection()
            cursor = conn.execute("SELECT * FROM mode")
            mode = list(cursor)[0][1]
            conn.close()

            updated_count = update_scan_count()

            global gamma
            gamma = 0.1

            return render_template(
                "index-new.html",
                upload=False,
                img=im_filename,
                omg=om_filename,
                gmg=gray_filename,
                vmg=vc_filename,
                mode=mode,
                updated_count=updated_count,
            )
        else:
            flash("No image selected")
            return render_template("index-new.html", upload=True)

    return render_template("index-new.html", upload=True)


@app.route("/old", methods=["GET", "POST"])
@timeit
def index_old():

    if "inputSubmit" in request.form:

        if "inputFile" in request.files:
            file = request.files["inputFile"]
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
            return render_template("index.html", upload=False, img=im_filename)
        else:
            flash("No image selected")
            return render_template("index.html", upload=True)

    return render_template("index.html", upload=True)


@app.route("/zeff", methods=["GET", "POST"])
@timeit
def zeff():

    if request.method == "POST":

        filename_orig = request.form["filename"]
        filename = (
            "static/original/" + os.path.basename(filename_orig).split(".")[0] + ".npy"
        )

        s = int(round(float(request.form["x1"])))
        t = int(round(float(request.form["y1"])))
        u = request.form["x2"]
        v = request.form["y2"]

        res, le, he = colorize(filename, all=True)

        a = le.flatten()
        b = he.flatten()
        c = np.log(a)
        d = np.log(b)
        R = c / d

        zeff = 5.7 * (R ** 2) - 7.4 * R + 8
        zeff1 = np.reshape(zeff, le.shape)
        I = res

        w = int(s) + int(u)
        y = int(t) + int(v)

        g = zeff1[int(t) : int(t) + 4, int(s) : int(s) + 4]

        return jsonify({"zeff": np.mean(g)})


@app.route("/constrast", methods=["GET", "POST"])
@timeit
def constrast():

    if request.method == "POST":
        global filename_global
        filename = (
            "static/colorized/"
            + os.path.basename(filename_global).split(".")[0]
            + ".png"
        )

        savePath = contrast(filename)

        return jsonify({"img": savePath})


@app.route("/rgbtohsi", methods=["GET", "POST"])
@timeit
def rgbtohsi():

    if request.method == "POST":

        filename = request.form["filename"]

        img = cv2.imread(filename)

        savePath = "static/hsi/" + os.path.basename(filename)

        img_hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imwrite(savePath, img_hsi)

        return jsonify({"img": savePath})

@app.route("/rgb", methods=["GET", "POST"])
@timeit
def rgb():

    if request.method == "POST":
        if "saveMode" in request.form:
            save_mode = request.form["saveMode"]

            if save_mode == "true":
                update_mode("pseudo-mode")

        global filename_global
        global brns_processing

        noobj_path = get_current_noobj_file_path()

        brns_processing = BRNSProcessing(filename_global, noobj_path=noobj_path)
        img = brns_processing.pc_img

        im_filename = (
            "static/colorized/"
            + os.path.basename(filename_global).split(".")[0]
            + ".png"
        )
        matplotlib.image.imsave(im_filename, img)
        return jsonify({"img": im_filename})


@app.route("/gray", methods=["GET", "POST"])
@timeit
def gray():

    if request.method == "POST":
        if "saveMode" in request.form:
            save_mode = request.form["saveMode"]

            if save_mode == "true":
                update_mode("grayscale-mode")

        global filename_global
        filename = filename_global
        img = colorize(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filename = os.path.basename(filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        im_filename = "static/grayscale/" + filename.split(".")[0] + ".png"
        cv2.imwrite(im_filename, img_gray)
        return jsonify({"img": im_filename})


@app.route("/hsi", methods=["GET", "POST"])
@timeit
def hsi():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        hsi_img = brns_processing.genHSIImg()
        im_filename = "static/hsi/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis("off")
        plt.imshow(hsi_img)
        plt.savefig(im_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})


@app.route("/cc", methods=["GET", "POST"])
@timeit
def cc():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        cc_img = brns_processing.genCCImg()
        filename_img = (
            "static/colorized/" + os.path.basename(filename).split(".")[0] + ".png"
        )
        im_real = cv2.imread(filename_img)
        shape_tuple = (im_real.shape[0], im_real.shape[1])
        cc_img = cv2.resize(cc_img, shape_tuple)
        im_filename = "static/cc/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis("off")
        plt.imshow(cc_img)
        plt.savefig(im_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})


@app.route("/inv", methods=["GET", "POST"])
@timeit
def inv():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        inv_img = brns_processing.genInvImg()
        im_filename = "static/inv/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis("off")
        plt.imshow(inv_img)
        plt.savefig(im_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})


@app.route("/obj", methods=["GET", "POST"])
@timeit
def obj():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        obj_img = brns_processing.genOvsBImg()
        im_filename = "static/obj/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis("off")
        plt.imshow(obj_img)
        plt.savefig(im_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})


@app.route("/om", methods=["GET", "POST"])
@timeit
def om():

    if request.method == "POST":
        if "saveMode" in request.form:
            save_mode = request.form["saveMode"]

            if save_mode == "true":
                update_mode("om-mode")

        global filename_global
        filename = filename_global
        global brns_processing
        om_img = brns_processing.genOMImg()
        im_filename = "static/om/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis("off")
        plt.imshow(om_img)
        plt.savefig(im_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})


@app.route("/im", methods=["GET", "POST"])
@timeit
def im():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        im_img = brns_processing.genIMImg()
        im_filename = "static/im/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis("off")
        plt.imshow(im_img)
        plt.savefig(im_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})


@app.route("/vcminus", methods=["GET", "POST"])
@timeit
def vcminus():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        vcminus_img = brns_processing.genVCminus()
        im_filename = (
            "static/vcminus/" + os.path.basename(filename.split(".")[0]) + ".png"
        )
        plt.axis("off")
        plt.imshow(vcminus_img)
        plt.savefig(im_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})


@app.route("/vcplus", methods=["GET", "POST"])
@timeit
def vcplus():

    if request.method == "POST":
        if "saveMode" in request.form:
            save_mode = request.form["saveMode"]

            if save_mode == "true":
                update_mode("vcplus-mode")

        global filename_global
        filename = filename_global
        global brns_processing
        vcplus_img = brns_processing.genVCplus()
        im_filename = (
            "static/vcplus/" + os.path.basename(filename.split(".")[0]) + ".png"
        )
        plt.axis("off")
        plt.imshow(vcplus_img)
        plt.savefig(im_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename})


@app.route("/gamma", methods=["GET", "POST"])
@timeit
def gamma():

    if request.method == "POST":
        global filename_global
        filename = (
            "static/colorized/"
            + os.path.basename(filename_global).split(".")[0]
            + ".png"
        )
        gamma = request.form["gamma"]
        savePath = gamma_correction(gamma, filename)
        return jsonify({"img": savePath, "gamma": gamma[:3]})


@app.route("/gamma-update", methods=["GET", "POST"])
@timeit
def gamma_update():

    if request.method == "POST":
        global filename_global
        global gamma
        filename = (
            "static/colorized/"
            + os.path.basename(filename_global).split(".")[0]
            + ".png"
        )
        savePath = gamma_correction(gamma, filename)
        gamma = gamma + 0.1
        return jsonify({"img": savePath, "gamma": gamma})


@app.route("/ve", methods=["GET", "POST"])
@timeit
def ve():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing

        ve_val = round(float(request.form["ve"]), 2)

        ve_img = brns_processing.genVEImg(ve_val)
        im_filename = "static/ve/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis("off")
        plt.imshow(ve_img)
        plt.savefig(im_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename, "ve": round(float(ve_val), 2)})


@app.route("/vd", methods=["GET", "POST"])
@timeit
def vd():

    if request.method == "POST":
        global filename_global
        filename = filename_global
        global brns_processing
        vd_val = float(request.form["vd"])

        vd_img = brns_processing.genVDImg(vd_val)
        im_filename = "static/vd/" + os.path.basename(filename.split(".")[0]) + ".png"
        plt.axis("off")
        plt.imshow(vd_img)
        plt.savefig(im_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        return jsonify({"img": im_filename, "vd": round(float(vd_val), 2)})


@app.route("/predict", methods=["POST"])
@timeit
def predict():

    if request.method == "POST":
        filename = request.form["filename"]
        im = cv2.imread(filename)

        s = int(round(float(request.form["x1"])))
        t = int(round(float(request.form["y1"])))
        u = request.form["x2"]
        v = request.form["y2"]

        w = int(s) + int(u)
        y = int(t) + int(v)

        g = im[int(t) : int(t) + 64, int(s) : int(s) + 64]
        file_save_name = (
            "static/nn/"
            + filename.split("/")[-1].split(".")[0]
            + "_"
            + str(time.time())
            + ".jpg"
        )
        cv2.imwrite(file_save_name, g)

        prediction = predict_function(file_save_name)
        icon = "success"

        if prediction == "Explosive":
            icon = "error"

        return jsonify({"icon": icon, "prediction": prediction})


@app.route("/jaccard", methods=["POST"])
@timeit
def jaccard_index():

    if request.method == "POST":
        filename = request.form["filename"]
        im = cv2.imread(filename)

        s = int(round(float(request.form["x1"])))
        t = int(round(float(request.form["y1"])))
        u = request.form["x2"]
        v = request.form["y2"]

        g = im[int(t) : int(t) + 10, int(s) : int(s) + 10]
        file_save_name = (
            "static/jaccard/"
            + filename.split("/")[-1].split(".")[0]
            + "_"
            + str(time.time())
            + ".png"
        )
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

        return jsonify(
            {
                "icon": "success",
                "status": f"Type: {max_type}, with score: {max_score}",
                "max_type": max_type,
            }
        )


@app.route("/cosine", methods=["POST"])
@timeit
def cosine_similarity():

    if request.method == "POST":
        filename = request.form["filename"]
        im = cv2.imread(filename)

        s = int(round(float(request.form["x1"])))
        t = int(round(float(request.form["y1"])))
        u = request.form["x2"]
        v = request.form["y2"]

        g = im[int(t) : int(t) + 10, int(s) : int(s) + 10]
        file_save_name = (
            "static/cosine/"
            + filename.split("/")[-1].split(".")[0]
            + "_"
            + str(time.time())
            + ".png"
        )
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

        return jsonify(
            {
                "icon": "success",
                "status": f"Type: {max_type}, with score: {max_score}",
                "max_type": max_type,
            }
        )


@app.route("/download")
@timeit
def download():

    return send_from_directory("static", "model/vanilla-cnn-colored.pth")


if __name__ == "__main__":
    app.run()
    plt.close()
