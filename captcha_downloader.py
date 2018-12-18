from bs4 import BeautifulSoup
import glob
from PIL import Image
import wget
import os
current = 0
for i in range(100):
    fname = "captcha_"+str(current)+".gif"
    wget.download("http://www.sxceducation.net/WebAttn/Captcha.aspx",fname)
    current+=1

image_files = glob.glob("*.gif")
for image in image_files:
    im = Image.open(image)
    new_im = Image.new("RGB",im.size)
    new_im.paste(im)
    new_im.save(image.split('.')[0]+".jpg","JPEG")
    os.remove(image)

