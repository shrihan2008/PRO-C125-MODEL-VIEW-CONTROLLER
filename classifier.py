import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml

import pandas as pd
from PIL import Image

import os,ssl,time

import PIL.ImageOps
X=np.load("image.npz")['arr_0']

y=pd.read_csv("labels.csv")['labels']

classes=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0.58,train_size=3500,test_size=500)
X_train_scale=X_train/255.0
X_test_scale=X_test/255.0

classi=LogisticRegression(solver='saga',multi_class='multinomial').fit(X_train_scale,y_train)

def get_prediction(image):
    im_pill=Image.open(image)
    img=im_pill.convert('L')
    img_resize=img.resize((22,30),Image.ANTIALIAS)
    px_filter=20
    min_px=np.percentile(img_resize,px_filter)
    img_resize_inverted=np.clip(img_resize-min_px,0,255)
    max_px=np.max(img_resize)
    img_resize_inverted=np.asarray(img_resize_inverted)/max_px
    test_sample=np.array(img_resize_inverted).reshape(1,784)
    test_pred=classi.predict(test_sample)
    return test_pred[0]

