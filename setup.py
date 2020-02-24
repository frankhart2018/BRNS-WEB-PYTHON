import os

files = ["cc", "colorized", "contrast", "gamma_corrected", "grayscale", "hsi", "im", "inv", "obj", "om", "original", "resize", "vcminus", "vcplus", "vd", "ve", "jaccard", "model", "nn"]

for file in files:
    if not os.path.exists("static/" + file):
        os.mkdir("static/" + file)
