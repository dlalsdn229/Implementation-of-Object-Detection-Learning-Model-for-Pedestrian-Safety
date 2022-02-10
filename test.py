from PIL import Image
import os, glob, numpy as np
from tensorflow.keras.models import load_model

import tensorflow as tf

seed = 5
tf.random.set_seed(seed)
#tf.set_random_seed(seed)
np.random.seed(seed)

caltech_dir = './testset'


image_w = 64
image_h = 64

pixels = image_h * image_w * 3

X = []

filenames = []
files = glob.glob(caltech_dir+"/*/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h),3)
    data = np.asarray(img)

    filenames.append(f)
    X.append(data)

print(X)
X = np.array(X)
X = X.astype(float) / 255
model = load_model('./model/WB.model')

prediction = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0
for i in prediction:
    if i >= 0.5:
        print("해당 " + filenames[cnt].split("\\")[1] + filenames[cnt].split("\\")[2] + "  이미지는 No 로 추정됩니다." ,i)
    else :
        print("해당 " + filenames[cnt].split("\\")[1] + filenames[cnt].split("\\")[2] + "  이미지는 Yes 으로 추정됩니다." ,i)
    cnt += 1
