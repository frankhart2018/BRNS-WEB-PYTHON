# Example Python Program for contrast stretching
from PIL import Image
import os
import cv2

# Method to process the red band of the image
def normalizeRed(intensity):

    iI      = intensity
    minI    = 86
    maxI    = 230
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)

    return iO

# Method to process the green band of the image
def normalizeGreen(intensity):

    iI      = intensity
    minI    = 90
    maxI    = 225
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)

    return iO

# Method to process the blue band of the image
def normalizeBlue(intensity):

    iI      = intensity
    minI    = 100
    maxI    = 210
    minO    = 0
    maxO    = 255
    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)

    return iO

def contrast(img_src):

    # Create an image object
    imageObject     = Image.open(img_src)

    # Split the red, green and blue bands from the Image
    multiBands      = imageObject.split()

    # Apply point operations that does contrast stretching on each color band
    normalizedRedBand      = multiBands[0].point(normalizeRed)
    normalizedGreenBand    = multiBands[1].point(normalizeGreen)
    normalizedBlueBand     = multiBands[2].point(normalizeBlue)

    # Create a new image from the contrast stretched red, green and blue brands
    normalizedImage = Image.merge("RGB", (normalizedRedBand, normalizedGreenBand, normalizedBlueBand))

    savePath = "./static/contrast/" + os.path.basename(img_src)

    normalizedImage.save(savePath, "JPEG")

    return savePath
