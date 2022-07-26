import glob
from PIL import Image
maskkeys = glob.glob("mask/256/*.bmp")
img = Image.open(maskkeys[0])
print(maskkeys)
print(img)