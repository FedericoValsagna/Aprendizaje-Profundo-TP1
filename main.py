from time import time

from PIL import Image
import numpy as np
import os

from HopfieldNetwork import HopfieldNetwork
from ej1 import ej1
from ej2 import ej2
from image_processing import numpy_array_to_bmp, transform_images

def setup():
    np.set_printoptions(threshold=np.inf)

def main():
    setup()
    ej1()
    ej2()

main()


