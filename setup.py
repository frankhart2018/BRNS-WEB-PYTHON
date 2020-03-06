import os
import urllib.request
import progressbar

files = ["cc", "colorized", "contrast", "gamma_corrected", "grayscale", "hsi", "im", "inv", "obj", "om", "original", "resize", "vcminus", "vcplus", "vd", "ve", "jaccard", "model", "nn"]

print()
print("Step 1) Creating intermediary files.")

for file in files:
    if not os.path.exists("static/" + file):
        os.mkdir("static/" + file)

print("Step 1 completed.")

pbar = None

def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None

print()
print("Step 2) Downloading neural network trained model.")
url = "http://jncpasighat.edu.in/file/vanilla-cnn-colored.pth"
urllib.request.urlretrieve(url, os.path.join(os.getcwd(), 'static/model/vanilla-cnn-colored.pth'), show_progress)

print("Step 2 completed.")
